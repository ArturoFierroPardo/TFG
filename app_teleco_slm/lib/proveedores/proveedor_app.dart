/// Global app state provider.
library;

import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import '../modelos/modelos_chat.dart';
import '../modelos/metricas_inferencia.dart';
import '../servicios/servicio_bd.dart';
import '../servicios/servicio_llm.dart';
import '../servicios/servicio_configuracion.dart';
import '../utilidades/constantes.dart';
import '../utilidades/traducciones.dart';

enum ModelStatus { notFound, found, loading, loaded, error }
enum ActiveSource { none, localModel, remoteServer }

class AppProvider extends ChangeNotifier {
  final ServicioBD _db = ServicioBD();
  final ServicioLLM _llm = ServicioLLM();
  final ServicioConfiguracion _settings = ServicioConfiguracion();

  ModelStatus _modelStatus = ModelStatus.notFound;
  ActiveSource _activeSource = ActiveSource.none;
  List<Conversation> _conversations = [];
  Conversation? _currentConversation;
  List<ChatMessage> _messages = [];
  bool _isGenerating = false;
  String? _generatingConvId;
  StreamSubscription<String>? _streamSub;
  final List<InferenceMetrics> _metricsHistory = [];

  ModelStatus get modelStatus => _modelStatus;
  ActiveSource get activeSource => _activeSource;
  List<Conversation> get conversations => _conversations;
  Conversation? get currentConversation => _currentConversation;
  List<ChatMessage> get messages => _messages;
  bool get isGenerating => _isGenerating;
  bool get isCurrentChatGenerating =>
      _isGenerating && _generatingConvId == _currentConversation?.id;
  ServicioLLM get llm => _llm;
  ServicioConfiguracion get settings => _settings;
  bool get isModelLoaded => _modelStatus == ModelStatus.loaded;
  List<InferenceMetrics> get metricsHistory => _metricsHistory;

  bool get isDesktop => Platform.isWindows || Platform.isLinux || Platform.isMacOS;

  Future<void> initialize() async {
    await _settings.inicializar();
    await _loadConversations();

    final savedLang = _settings.idioma;
    if (savedLang.isNotEmpty) {
      Tr.setLang(savedLang);
    }

    if (isDesktop) {
      final path = _settings.rutaModelo;
      if (path.isNotEmpty && File(path).existsSync()) {
        await loadModel(path);
        return;
      }
      final exeDir = File(Platform.resolvedExecutable).parent.path;
      final dir = Directory(exeDir);
      try {
        final ggufFiles = dir.listSync().where(
          (f) => f.path.toLowerCase().endsWith('.gguf'),
        ).toList();
        if (ggufFiles.isNotEmpty) {
          await loadModel(ggufFiles.first.path);
        }
      } catch (_) {}
    } else {
      // Android: detect model and load it immediately
      final localModel = await ServicioLLM.findLocalModel();
      if (localModel != null) {
        await loadLocalModel(localModel);
        return;
      }
      final serverUrl = _settings.serverUrl;
      if (serverUrl.isNotEmpty) {
        await connectToServer(serverUrl);
      }
    }
  }

  // === Language ===

  void setLanguage(String lang) {
    Tr.setLang(lang);
    _settings.guardarIdioma(lang);
    notifyListeners();
  }

  // === Local Model (Windows) ===

  Future<void> loadModel(String path) async {
    _modelStatus = ModelStatus.loading;
    notifyListeners();
    final ok = await _llm.cargarModelo(path);
    if (ok) {
      _modelStatus = ModelStatus.loaded;
      _activeSource = ActiveSource.localModel;
      await _settings.guardarRutaModelo(path);
    } else {
      _modelStatus = ModelStatus.error;
    }
    notifyListeners();
  }

  // === Local Model (Android - llama_flutter_android) ===

  Future<void> loadLocalModel(String path) async {
    _modelStatus = ModelStatus.loading;
    notifyListeners();
    try {
      final ok = await _llm.loadLocalModel(path);
      if (ok) {
        _modelStatus = ModelStatus.loaded;
        _activeSource = ActiveSource.localModel;
      } else {
        _modelStatus = ModelStatus.error;
      }
    } catch (e) {
      print('Error loading local model: $e');
      _modelStatus = ModelStatus.error;
      _llm.unloadLocalModel();
    }
    notifyListeners();
  }

  void unloadLocalModel() {
    if (_activeSource != ActiveSource.localModel) return;
    _llm.unloadLocalModel();
    _modelStatus = ModelStatus.notFound;
    _activeSource = ActiveSource.none;
    notifyListeners();
  }

  // === Server (Windows & Android) ===

  Future<void> connectToServer(String url) async {
    _modelStatus = ModelStatus.loading;
    notifyListeners();
    final ok = await _llm.connectToServer(url);
    if (ok) {
      _modelStatus = ModelStatus.loaded;
      _activeSource = ActiveSource.remoteServer;
      await _settings.guardarServerUrl(url);
    } else {
      _modelStatus = ModelStatus.error;
    }
    notifyListeners();
  }

  void unloadModel() {
    if (_activeSource != ActiveSource.localModel) return;
    _llm.descargarModelo();
    _modelStatus = ModelStatus.notFound;
    _activeSource = ActiveSource.none;
    notifyListeners();
  }

  void disconnectServer() {
    if (_activeSource != ActiveSource.remoteServer) return;
    _llm.disconnectServer();
    _modelStatus = ModelStatus.notFound;
    _activeSource = ActiveSource.none;
    notifyListeners();
  }

  // === Conversations ===

  Future<void> _loadConversations() async {
    _conversations = await _db.obtenerConversaciones();
    notifyListeners();
  }

  Future<void> startNewChat() async {
    final conv = Conversation(title: Tr.get('newChat'));
    await _db.crearConversacion(conv);
    _currentConversation = conv;
    _messages = [];
    await _loadConversations();
    notifyListeners();
  }

  Future<void> openConversation(Conversation conv) async {
    _currentConversation = conv;
    _messages = await _db.obtenerMensajes(conv.id);
    notifyListeners();
  }

  Future<void> deleteConversation(String id) async {
    await _db.eliminarConversacion(id);
    if (_currentConversation?.id == id) {
      _currentConversation = null;
      _messages = [];
    }
    await _loadConversations();
    notifyListeners();
  }

  // === Chat ===

  Future<void> sendMessage(String content) async {
    if (content.trim().isEmpty || _isGenerating) return;
    if (_currentConversation == null) return;

    final convId = _currentConversation!.id;

    final userMsg = ChatMessage(
      conversationId: convId,
      role: 'user',
      content: content.trim(),
    );
    _messages.add(userMsg);
    await _db.insertarMensaje(userMsg);

    if (_messages.where((m) => m.role == 'user').length == 1) {
      final title = content.trim().length > 45
          ? '${content.trim().substring(0, 45)}...'
          : content.trim();
      _currentConversation = _currentConversation!.copyWith(
        title: title,
        updatedAt: DateTime.now(),
      );
      await _db.actualizarConversacion(_currentConversation!);
      await _loadConversations();
    }

    notifyListeners();

    _isGenerating = true;
    _generatingConvId = convId;
    notifyListeners();

    final assistantMsg = ChatMessage(
      conversationId: convId,
      role: 'assistant',
      content: '',
      isStreaming: true,
    );
    _messages.add(assistantMsg);
    notifyListeners();

    final buffer = StringBuffer();
    final startTime = DateTime.now();
    DateTime? firstTokenTime;
    int tokenCount = 0;
    final cpuBefore = ProcessInfo.currentRss;

    try {
      _streamSub = _llm
          .generarStream(
        systemPrompt: AppConstants.systemPrompt,
        mensajes: _messages.where((m) => m.role != 'system').toList(),
        maxTokens: AppConstants.maxTokensCeiling,
        temperature: AppConstants.temperature,
      )
          .listen(
        (token) {
          if (firstTokenTime == null) {
            firstTokenTime = DateTime.now();
          }
          tokenCount++;
          buffer.write(token);
          final idx = _messages.length - 1;
          _messages[idx] = _messages[idx].copyWith(content: buffer.toString());
          notifyListeners();
        },
        onDone: () async {
          _isGenerating = false;
          _generatingConvId = null;
          final endTime = DateTime.now();
          final totalMs = endTime.difference(startTime).inMilliseconds.toDouble();
          final firstTokenMs = firstTokenTime != null
              ? firstTokenTime!.difference(startTime).inMilliseconds.toDouble()
              : totalMs;
          final tokPerSec = totalMs > 0 ? (tokenCount / (totalMs / 1000)) : 0.0;

          final cpuAfter = ProcessInfo.currentRss;
          final cpuDelta = (cpuAfter - cpuBefore).abs();
          final cpuPercent = cpuDelta > 0 ? (cpuDelta / cpuAfter * 100).clamp(0, 100) : 0.0;

          String backendName = 'unknown';
          if (_activeSource == ActiveSource.localModel && !isDesktop) {
            backendName = 'local (Android)';
          } else if (_activeSource == ActiveSource.localModel && isDesktop) {
            backendName = 'llama-server (local)';
          } else if (_activeSource == ActiveSource.remoteServer) {
            backendName = 'remote server';
          }

          _metricsHistory.insert(0, InferenceMetrics(
            timestamp: startTime,
            userPrompt: content.length > 60 ? '${content.substring(0, 60)}...' : content,
            tokensGenerated: tokenCount,
            timeToFirstTokenMs: firstTokenMs,
            totalTimeMs: totalMs,
            tokensPerSecond: tokPerSec,
            cpuUsagePercent: cpuPercent.toDouble(),
            backend: backendName,
          ));

          final finalMsg = _messages.last.copyWith(
            content: buffer.toString(),
            isStreaming: false,
          );
          _messages[_messages.length - 1] = finalMsg;
          await _db.insertarMensaje(finalMsg);
          _currentConversation = _currentConversation!.copyWith(
            updatedAt: DateTime.now(),
          );
          await _db.actualizarConversacion(_currentConversation!);
          await _loadConversations();
          notifyListeners();
        },
        onError: (e) {
          _isGenerating = false;
          _generatingConvId = null;
          _messages[_messages.length - 1] = _messages.last.copyWith(
            content: 'Error: $e',
            isStreaming: false,
          );
          notifyListeners();
        },
      );
    } catch (e) {
      _isGenerating = false;
      _generatingConvId = null;
      notifyListeners();
    }
  }

  Future<void> stopGeneration() async {
    _streamSub?.cancel();
    await _llm.detenerGeneracion();
    _isGenerating = false;
    _generatingConvId = null;
    if (_messages.isNotEmpty && _messages.last.isStreaming) {
      final currentContent = _messages.last.content;
      final stoppedMsg = _messages.last.copyWith(
        content: currentContent.isEmpty
            ? '[Respuesta detenida]'
            : currentContent,
        isStreaming: false,
      );
      _messages[_messages.length - 1] = stoppedMsg;
      // Save to DB so it persists after crash/restart
      await _db.insertarMensaje(stoppedMsg);
    }
    notifyListeners();
  }

  // === Settings ===

  Future<void> clearAllHistory() async {
    await _db.borrarTodoElHistorial();
    _conversations = [];
    _currentConversation = null;
    _messages = [];
    notifyListeners();
  }
}