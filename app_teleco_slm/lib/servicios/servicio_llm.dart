/// LLM inference service.
///
/// Windows: launches llama-server.exe locally via HTTP.
/// Android local: uses llama_flutter_android (llama.cpp b8201, Qwen3 compatible).
/// Android/Windows remote: connects to a remote llama-server via HTTP.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:llama_flutter_android/llama_flutter_android.dart' as llama;
import 'package:path_provider/path_provider.dart';
import '../modelos/modelos_chat.dart';
import '../utilidades/constantes.dart';

enum LLMBackend { none, localServer, remoteServer, llamaLocal }

class ServicioLLM {
  Process? _proceso;
  bool _estaCargado = false;
  bool _estaCargando = false;
  bool _esMock = false;
  String? _errorCarga;
  String _serverUrl = 'http://localhost:8089';
  final int _puerto = 8089;
  LLMBackend _backend = LLMBackend.none;

  // llama_flutter_android
  llama.LlamaController? _llamaController;
  String? _localModelPath;
  StreamSubscription<String>? _tokenSub;

  // Cancel support for HTTP streaming
  HttpClient? _activeHttpClient;
  bool _cancelRequested = false;

  // RAM info from last check
  int? _lastFreeRamMB;

  bool get estaCargado => _estaCargado;
  bool get estaCargando => _estaCargando;
  bool get esMock => _esMock;
  String? get errorCarga => _errorCarga;
  String get serverUrl => _serverUrl;
  LLMBackend get backend => _backend;
  String? get localModelPath => _localModelPath;
  int? get lastFreeRamMB => _lastFreeRamMB;

  bool get _isDesktop => Platform.isWindows || Platform.isLinux || Platform.isMacOS;

  // =========================================================================
  // RAM CHECK (Android - llama_flutter_android detectGpu)
  // =========================================================================

  /// Returns free RAM in MB, or null if detection fails.
  Future<int?> checkAvailableRam() async {
    try {
      final controller = llama.LlamaController();
      final gpu = await controller.detectGpu();
      final freeRamMB = gpu.freeRamBytes ~/ (1024 * 1024);
      _lastFreeRamMB = freeRamMB;
      print('RAM check: ${freeRamMB} MB free');
      await controller.dispose();
      return freeRamMB;
    } catch (e) {
      print('RAM check failed: $e');
      return null;
    }
  }

  // =========================================================================
  // LOCAL MODEL (Android - llama_flutter_android)
  // =========================================================================

  /// Minimum free RAM in MB required to attempt model loading.
  static const int _minFreeRamMB = 1800;

  Future<bool> loadLocalModel(String modelPath) async {
    _estaCargando = true;
    _errorCarga = null;

    try {
      if (!File(modelPath).existsSync()) {
        _errorCarga = 'Archivo del modelo no encontrado:\n$modelPath';
        _estaCargando = false;
        return false;
      }

      // Check file size
      final fileSizeMB = File(modelPath).lengthSync() ~/ (1024 * 1024);
      print('Model file size: ${fileSizeMB} MB');

      // Check available RAM before loading
      final freeRamMB = await checkAvailableRam();
      if (freeRamMB != null && freeRamMB < _minFreeRamMB) {
        _errorCarga = 'RAM insuficiente para cargar el modelo.\n'
            'Disponible: ${freeRamMB} MB — Necesario: ~${_minFreeRamMB} MB.\n'
            'Cierra otras apps e inténtalo de nuevo, o usa un modelo más pequeño.';
        _estaCargando = false;
        return false;
      }

      print('Loading model with llama_flutter_android: $modelPath');
      print('Free RAM: ${freeRamMB ?? "unknown"} MB');

      _llamaController = llama.LlamaController();

      // Dynamic threads: half of CPU cores, clamped 2-4
      int nThreads = (Platform.numberOfProcessors ~/ 2).clamp(2, 4);
      try {
        final gpu = await _llamaController!.detectGpu();
        print('GPU: ${gpu.gpuName}, Vulkan: ${gpu.vulkanSupported}, '
            'free RAM: ${gpu.freeRamBytes ~/ (1024 * 1024)} MB, threads: $nThreads');
      } catch (e) {
        print('GPU detection info failed: $e');
      }

      // gpuLayers=0 (CPU only) — Vulkan on mobile GPUs is unreliable
      await _llamaController!.loadModel(
        modelPath: modelPath,
        threads: nThreads,
        contextSize: 2048,
        gpuLayers: 0,
      );

      _localModelPath = modelPath;
      _estaCargado = true;
      _esMock = false;
      _backend = LLMBackend.llamaLocal;
      _estaCargando = false;
      print('Model loaded successfully');
      return true;
    } catch (e) {
      _errorCarga = 'Error al cargar el modelo: $e\n'
          'Es posible que no haya suficiente RAM.\n'
          'RAM libre: ${_lastFreeRamMB ?? "desconocida"} MB';
      _estaCargando = false;
      _llamaController = null;
      return false;
    }
  }

  void unloadLocalModel() {
    try {
      _llamaController?.dispose();
    } catch (_) {}
    _llamaController = null;
    _localModelPath = null;
    if (_backend == LLMBackend.llamaLocal) {
      _estaCargado = false;
      _backend = LLMBackend.none;
    }
  }

  static Future<String> getModelDirectory() async {
    final dir = await getApplicationDocumentsDirectory();
    final modelDir = Directory('${dir.path}/models');
    if (!await modelDir.exists()) {
      await modelDir.create(recursive: true);
    }
    return modelDir.path;
  }

  static Future<String?> findLocalModel() async {
    final modelDir = await getModelDirectory();
    final dir = Directory(modelDir);
    try {
      final files = dir.listSync().where(
        (f) => f.path.toLowerCase().endsWith('.gguf'),
      ).toList();
      if (files.isNotEmpty) return files.first.path;
    } catch (_) {}
    return null;
  }

  // =========================================================================
  // CONNECTION (remote server)
  // =========================================================================

  Future<bool> connectToServer(String url) async {
    _estaCargando = true;
    _errorCarga = null;

    var cleanUrl = url.trim();
    if (!cleanUrl.startsWith('http')) {
      cleanUrl = 'http://$cleanUrl';
    }
    if (cleanUrl.endsWith('/')) {
      cleanUrl = cleanUrl.substring(0, cleanUrl.length - 1);
    }

    try {
      final client = HttpClient();
      client.connectionTimeout = const Duration(seconds: 10);
      final request = await client.getUrl(Uri.parse('$cleanUrl/health'));
      final response = await request.close();
      final body = await response.transform(utf8.decoder).join();
      client.close();

      if (response.statusCode == 200 && body.contains('ok')) {
        _serverUrl = cleanUrl;
        _estaCargado = true;
        _esMock = false;
        _backend = LLMBackend.remoteServer;
        _estaCargando = false;
        print('Connected to server: $cleanUrl');
        return true;
      } else {
        _errorCarga = 'El servidor no respondió correctamente.';
        _estaCargando = false;
        return false;
      }
    } on SocketException catch (e) {
      _errorCarga = 'No se pudo conectar a $cleanUrl\n'
          'Verifica que el servidor está encendido y la IP es correcta.\n'
          '(${e.message})';
      _estaCargando = false;
      return false;
    } on HttpException catch (e) {
      _errorCarga = 'Error HTTP: ${e.message}';
      _estaCargando = false;
      return false;
    } catch (e) {
      _errorCarga = 'Error de conexión: $e';
      _estaCargando = false;
      return false;
    }
  }

  void disconnectServer() {
    _serverUrl = 'http://localhost:8089';
    if (_backend == LLMBackend.remoteServer) {
      _estaCargado = false;
      _backend = LLMBackend.none;
    }
  }

  // =========================================================================
  // LOAD MODEL (Windows - local llama-server)
  // =========================================================================

  Future<bool> cargarModelo(String rutaModelo) async {
    if (!_isDesktop) {
      _errorCarga = 'Usa loadLocalModel() en Android.';
      return false;
    }
    if (_estaCargando) return false;
    _estaCargando = true;
    _errorCarga = null;

    try {
      if (!File(rutaModelo).existsSync()) {
        _errorCarga = 'Modelo no encontrado en:\n$rutaModelo';
        _estaCargando = false;
        return false;
      }

      final rutaServer = _buscarServer();
      if (rutaServer == null) {
        _errorCarga = 'llama-server.exe no encontrado.\n'
            'Asegúrate de que está en la misma carpeta que la app.';
        _estaCargando = false;
        return false;
      }

      _proceso?.kill();
      final nThreads = (Platform.numberOfProcessors ~/ 2).clamp(1, 8);

      try {
        _proceso = await Process.start(rutaServer, [
          '-m', rutaModelo,
          '--port', '$_puerto',
          '-ngl', '0',
          '-c', '${AppConstants.contextWindow}',
          '-t', '$nThreads',
          '--jinja',
          '--reasoning-format', 'deepseek',
        ]);

        print('llama-server launched on port $_puerto (with --jinja --reasoning-format deepseek)');

        _proceso!.stderr.transform(utf8.decoder).listen((data) {
          if (data.contains('listening') || data.contains('model loaded') || data.contains('error')) {
            print('[server] $data');
          }
        });

        final listo = await _esperarServidor(60);
        if (listo) {
          _estaCargado = true;
          _esMock = false;
          _backend = LLMBackend.localServer;
          _serverUrl = 'http://localhost:$_puerto';
          print('Server ready at $_serverUrl');
        } else {
          _proceso?.kill();
          _proceso = null;
          print('Retrying without --jinja flags...');

          _proceso = await Process.start(rutaServer, [
            '-m', rutaModelo,
            '--port', '$_puerto',
            '-ngl', '0',
            '-c', '${AppConstants.contextWindow}',
            '-t', '$nThreads',
          ]);

          _proceso!.stderr.transform(utf8.decoder).listen((data) {
            if (data.contains('listening') || data.contains('model loaded') || data.contains('error')) {
              print('[server-fallback] $data');
            }
          });

          final listoFallback = await _esperarServidor(60);
          if (listoFallback) {
            _estaCargado = true;
            _esMock = false;
            _backend = LLMBackend.localServer;
            _serverUrl = 'http://localhost:$_puerto';
            print('Server ready (fallback mode) at $_serverUrl');
          } else {
            _proceso?.kill();
            _proceso = null;
            _errorCarga = 'El servidor no respondió a tiempo.\n'
                'Puede que el modelo sea demasiado grande para la RAM.';
            _estaCargando = false;
            return false;
          }
        }
      } on ProcessException catch (e) {
        _errorCarga = 'No se pudo lanzar llama-server.exe\n'
            'Puede que falten DLLs (llama-server-impl.dll, ggml-*.dll).\n'
            'Copia TODOS los .dll del zip de llama.cpp junto al .exe.\n'
            '(${e.message})';
        _estaCargando = false;
        return false;
      } catch (e) {
        print('Error launching server: $e');
        _esMock = true;
        _estaCargado = true;
      }

      _estaCargando = false;
      return true;
    } catch (e) {
      _errorCarga = 'Error: $e';
      _estaCargando = false;
      return false;
    }
  }

  String? _buscarServer() {
    final rutas = [
      '${File(Platform.resolvedExecutable).parent.path}${Platform.pathSeparator}llama-server.exe',
      'C:\\Users\\artur\\Desktop\\TelecoSLM_Instalador\\llama-server.exe',
      'llama-server.exe',
    ];
    for (final ruta in rutas) {
      if (File(ruta).existsSync()) {
        print('llama-server found: $ruta');
        return ruta;
      }
    }
    return null;
  }

  Future<bool> _esperarServidor(int maxSegundos) async {
    final client = HttpClient();
    for (var i = 0; i < maxSegundos * 2; i++) {
      try {
        final request = await client.getUrl(Uri.parse('http://localhost:$_puerto/health'));
        final response = await request.close();
        final body = await response.transform(utf8.decoder).join();
        if (response.statusCode == 200 && body.contains('ok')) {
          client.close();
          return true;
        }
      } catch (_) {}
      await Future.delayed(const Duration(milliseconds: 500));
    }
    client.close();
    return false;
  }

  void descargarModelo() {
    _proceso?.kill();
    _proceso = null;
    if (_backend == LLMBackend.localServer) {
      _estaCargado = false;
      _backend = LLMBackend.none;
    }
    _esMock = false;
  }

  /// Stops the current generation and resets native state.
  Future<void> detenerGeneracion() async {
    _cancelRequested = true;

    if (_backend == LLMBackend.llamaLocal && _llamaController != null) {
      try {
        _llamaController!.stop();
      } catch (_) {}
      // Wait briefly for native code to finish, then reset context
      await Future.delayed(const Duration(milliseconds: 500));
      try {
        await _llamaController!.clearContext();
      } catch (_) {}
    }

    _tokenSub?.cancel();
    _tokenSub = null;

    try {
      _activeHttpClient?.close(force: true);
    } catch (_) {}
    _activeHttpClient = null;
  }

  // =========================================================================
  // GENERATION - routes to correct backend
  // =========================================================================

  Stream<String> generarStream({
    required String systemPrompt,
    required List<ChatMessage> mensajes,
    int maxTokens = 4096,
    double temperature = 0.6,
  }) async* {
    _cancelRequested = false;

    if (_esMock || !_estaCargado) {
      yield* _streamMock(mensajes.last.content);
      return;
    }

    if (_backend == LLMBackend.llamaLocal) {
      yield* _generarConLlama(
        systemPrompt: systemPrompt,
        mensajes: mensajes,
        maxTokens: maxTokens,
        temperature: temperature,
      );
    } else {
      yield* _generarConHTTP(
        systemPrompt: systemPrompt,
        mensajes: mensajes,
        maxTokens: maxTokens,
        temperature: temperature,
      );
    }
  }

  // =========================================================================
  // LOCAL GENERATION (llama_flutter_android - uses generateChat with chatml)
  // Thinking mode ON for quality; <think> blocks filtered from output
  // =========================================================================

  /// Rough token estimate: ~3.5 chars per token for Spanish text
  static int _estimateTokens(String text) => (text.length / 3.5).ceil();

  /// Context budget: 2048 total, reserve 500 for thinking + 400 for response
  static const int _maxPromptTokens = 1100;

  Stream<String> _generarConLlama({
    required String systemPrompt,
    required List<ChatMessage> mensajes,
    int maxTokens = 4096,
    double temperature = 0.6,
  }) async* {
    if (_llamaController == null) {
      yield '[Error: modelo no cargado]';
      return;
    }

    // Clear KV cache before each generation
    try {
      await _llamaController!.clearContext();
    } catch (_) {}

    // Build system message (thinking mode stays ON for better math/formulas)
    final systemContent = _limpiarTexto(systemPrompt);

    // On Android, only send system + last user message to minimize memory
    // pressure and avoid native crashes. Conversation history is visual only.
    final chatMessages = <llama.ChatMessage>[];
    chatMessages.add(llama.ChatMessage(
      role: 'system',
      content: systemContent,
    ));

    // Find the last user message
    final lastUserMsg = mensajes.lastWhere(
      (m) => m.role == 'user',
      orElse: () => mensajes.last,
    );
    chatMessages.add(llama.ChatMessage(
      role: 'user',
      content: _limpiarTexto(lastUserMsg.content),
    ));

    final controller = StreamController<String>();
    _tokenSub?.cancel();

    // State for filtering <think>...</think> blocks
    bool insideThink = false;
    final tagBuffer = StringBuffer();

    try {
      _tokenSub = _llamaController!.generateChat(
        messages: chatMessages,
        template: 'chatml',
        maxTokens: maxTokens,
        temperature: temperature,
        topP: 0.9,
        topK: 40,
        repeatPenalty: 1.1,
      ).listen(
        (token) {
          if (_cancelRequested) {
            _tokenSub?.cancel();
            if (!controller.isClosed) controller.close();
            return;
          }
          if (token.isEmpty) return;

          tagBuffer.write(token);
          final buf = tagBuffer.toString();

          if (insideThink) {
            if (buf.contains('</think>')) {
              final after = _cleanToken(buf.split('</think>').last);
              tagBuffer.clear();
              insideThink = false;
              if (after.isNotEmpty) controller.add(after);
            } else if (tagBuffer.length > 10000) {
              tagBuffer.clear(); // safety limit
            }
          } else {
            if (buf.contains('<think>')) {
              final before = _cleanToken(buf.split('<think>').first);
              final rest = buf.substring(buf.indexOf('<think>') + 7);
              if (rest.contains('</think>')) {
                final after = _cleanToken(rest.split('</think>').last);
                tagBuffer.clear();
                if (before.isNotEmpty) controller.add(before);
                if (after.isNotEmpty) controller.add(after);
              } else {
                tagBuffer.clear();
                insideThink = true;
                if (before.isNotEmpty) controller.add(before);
              }
            } else if (buf.contains('<')) {
              if (tagBuffer.length > 20) {
                final cleaned = _cleanToken(buf);
                if (cleaned.isNotEmpty) controller.add(cleaned);
                tagBuffer.clear();
              }
            } else {
              final cleaned = _cleanToken(buf);
              if (cleaned.isNotEmpty) controller.add(cleaned);
              tagBuffer.clear();
            }
          }
        },
        onDone: () {
          if (!insideThink && tagBuffer.isNotEmpty) {
            controller.add(tagBuffer.toString());
          }
          tagBuffer.clear();
          if (!controller.isClosed) controller.close();
        },
        onError: (e) {
          if (!controller.isClosed) {
            controller.add('[Error: $e]');
            controller.close();
          }
        },
      );
    } catch (e) {
      controller.add('[Error: $e]');
      controller.close();
    }

    bool emittedAny = false;
    final visibleBuffer = StringBuffer();
    bool timedOut = false;

    try {
      // Timeout: if no tokens arrive in 60 seconds, consider it stuck
      await for (final token in controller.stream.timeout(
        const Duration(seconds: 60),
        onTimeout: (sink) {
          timedOut = true;
          sink.close();
        },
      )) {
        if (_cancelRequested) break;
        visibleBuffer.write(token);
        emittedAny = true;
        yield token;
      }
    } catch (_) {
      // Timeout or stream error
    }

    // Cancel any ongoing generation
    _tokenSub?.cancel();
    _tokenSub = null;
    try { _llamaController?.stop(); } catch (_) {}

    // Check if visible output is just whitespace
    final visibleText = visibleBuffer.toString().trim();
    if ((visibleText.isEmpty || !emittedAny || timedOut) && !_cancelRequested) {
      // Auto-retry with /no_think to force a visible response
      yield '[Reintentando sin modo pensamiento]\n\n';

      try {
        await _llamaController!.clearContext();
      } catch (_) {}

      final retryMessages = <llama.ChatMessage>[];
      retryMessages.add(llama.ChatMessage(
        role: 'system',
        content: '${_limpiarTexto(systemPrompt)}\n/no_think',
      ));
      retryMessages.add(llama.ChatMessage(
        role: 'user',
        content: _limpiarTexto(lastUserMsg.content),
      ));

      final retryController = StreamController<String>();
      _tokenSub?.cancel();

      try {
        _tokenSub = _llamaController!.generateChat(
          messages: retryMessages,
          template: 'chatml',
          maxTokens: maxTokens,
          temperature: temperature,
          topP: 0.9,
          topK: 40,
          repeatPenalty: 1.1,
        ).listen(
          (token) {
            if (_cancelRequested) {
              _tokenSub?.cancel();
              if (!retryController.isClosed) retryController.close();
              return;
            }
            if (token.isNotEmpty) {
              final cleaned = _cleanToken(token);
              if (cleaned.isNotEmpty) retryController.add(cleaned);
            }
          },
          onDone: () {
            if (!retryController.isClosed) retryController.close();
          },
          onError: (e) {
            if (!retryController.isClosed) retryController.close();
          },
        );
      } catch (_) {
        retryController.close();
      }

      bool retryEmitted = false;
      try {
        await for (final token in retryController.stream.timeout(
          const Duration(seconds: 60),
          onTimeout: (sink) => sink.close(),
        )) {
          if (_cancelRequested) break;
          retryEmitted = true;
          yield token;
        }
      } catch (_) {}

      if (!retryEmitted && !_cancelRequested) {
        yield '\n\n⚠️ No se pudo generar respuesta. Abre un nuevo chat e inténtalo de nuevo.';
      }

      _tokenSub?.cancel();
      _tokenSub = null;
    }
  }

  // =========================================================================
  // HTTP GENERATION (for local server + remote server)
  // =========================================================================

  Stream<String> _generarConHTTP({
    required String systemPrompt,
    required List<ChatMessage> mensajes,
    int maxTokens = 4096,
    double temperature = 0.6,
  }) async* {
    final mensajesApi = <Map<String, String>>[];
    mensajesApi.add({'role': 'system', 'content': _limpiarTexto(systemPrompt)});

    final recientes = mensajes.length > AppConstants.maxContextMessages
        ? mensajes.sublist(mensajes.length - AppConstants.maxContextMessages)
        : mensajes;

    for (final msg in recientes) {
      if (msg.role == 'system') continue;
      mensajesApi.add({'role': msg.role, 'content': _limpiarTexto(msg.content)});
    }

    final bodyMap = {
      'messages': mensajesApi,
      'max_tokens': maxTokens,
      'temperature': temperature,
      'top_p': 0.95,
      'top_k': 20,
      'min_p': 0.0,
      'repeat_penalty': 1.1,
      'stream': true,
      'stop': ['<|im_end|>', '<|im_start|>'],
    };

    final bodyBytes = utf8.encode(jsonEncode(bodyMap));

    final client = HttpClient();
    _activeHttpClient = client;

    // Think filter state for HTTP stream
    bool insideThink = false;
    final tagBuffer = StringBuffer();
    bool emittedAny = false;

    try {
      final request = await client.postUrl(
        Uri.parse('$_serverUrl/v1/chat/completions'),
      );
      request.headers.set('Content-Type', 'application/json; charset=utf-8');
      request.add(bodyBytes);

      final response = await request.close();

      if (response.statusCode != 200) {
        final errorBody = await response.transform(utf8.decoder).join();
        try {
          final errorJson = jsonDecode(errorBody);
          final msg = errorJson['error']?['message'] ?? errorBody;
          yield '[Error del servidor: $msg]';
        } catch (_) {
          yield '[Error del servidor: código ${response.statusCode}]';
        }
        return;
      }

      await for (final chunk in response.transform(utf8.decoder)) {
        if (_cancelRequested) break;

        final lineas = chunk.split('\n');
        for (final linea in lineas) {
          if (_cancelRequested) break;
          if (!linea.startsWith('data: ')) continue;
          final datos = linea.substring(6).trim();
          if (datos == '[DONE]') break;

          try {
            final json = jsonDecode(datos);
            final choices = json['choices'] as List?;
            if (choices != null && choices.isNotEmpty) {
              final delta = choices[0]['delta'] as Map?;
              if (delta != null) {
                // Skip reasoning_content (deepseek format)
                final contenido = delta['content'] as String?;
                if (contenido == null || contenido.isEmpty) continue;

                // Filter <think>...</think> blocks
                tagBuffer.write(contenido);
                final buf = tagBuffer.toString();

                if (insideThink) {
                  if (buf.contains('</think>')) {
                    final after = buf.split('</think>').last;
                    tagBuffer.clear();
                    insideThink = false;
                    if (after.isNotEmpty) { yield after; emittedAny = true; }
                  } else if (tagBuffer.length > 10000) {
                    tagBuffer.clear();
                  }
                } else {
                  if (buf.contains('<think>')) {
                    final before = buf.split('<think>').first;
                    final rest = buf.substring(buf.indexOf('<think>') + 7);
                    if (rest.contains('</think>')) {
                      final after = rest.split('</think>').last;
                      tagBuffer.clear();
                      if (before.isNotEmpty) { yield before; emittedAny = true; }
                      if (after.isNotEmpty) { yield after; emittedAny = true; }
                    } else {
                      tagBuffer.clear();
                      insideThink = true;
                      if (before.isNotEmpty) { yield before; emittedAny = true; }
                    }
                  } else if (buf.contains('<')) {
                    if (tagBuffer.length > 20) {
                      yield buf; emittedAny = true;
                      tagBuffer.clear();
                    }
                  } else {
                    yield buf; emittedAny = true;
                    tagBuffer.clear();
                  }
                }
              }
            }
          } catch (_) {}
        }
      }

      // Flush remaining buffer
      if (!insideThink && tagBuffer.isNotEmpty) {
        yield tagBuffer.toString();
        emittedAny = true;
      }
      tagBuffer.clear();

      if (!emittedAny && !_cancelRequested) {
        yield '⚠️ El modelo ha pensado pero no ha generado respuesta visible. '
            'Prueba a reformular la pregunta o abre un nuevo chat.';
      }
    } on SocketException catch (e) {
      if (!_cancelRequested) {
        yield '[Error de conexión: ${e.message}]';
      }
    } on HttpException catch (e) {
      if (!_cancelRequested) {
        yield '[Error HTTP: ${e.message}]';
      }
    } catch (e) {
      if (!_cancelRequested) {
        yield '[Error: $e]';
      }
    } finally {
      _activeHttpClient = null;
      try { client.close(); } catch (_) {}
    }
  }

  // =========================================================================
  // UTILS
  // =========================================================================

  String _limpiarTexto(String texto) {
    try {
      final bytes = utf8.encode(texto);
      return utf8.decode(bytes, allowMalformed: true);
    } catch (_) {
      return texto.replaceAll(RegExp(r'[^\x00-\x7F]'), '?');
    }
  }

  /// Cleans a single token/chunk: removes ChatML artifacts like "assistant"
  static String _cleanToken(String token) {
    var clean = token;
    clean = clean.replaceAll(RegExp(r'^assistant\s*', caseSensitive: false), '');
    clean = clean.replaceAll('<|im_start|>', '');
    clean = clean.replaceAll('<|im_end|>', '');
    clean = clean.replaceAll(RegExp(r'<\|im_start\|>.*?\n?'), '');
    return clean;
  }

  /// Cleans raw model output: strips thinking blocks, role labels, and
  /// special tokens that may leak through.
  static String limpiarSalida(String texto) {
    var limpio = texto;

    // Remove <think>...</think> blocks (including multiline)
    limpio = limpio.replaceAll(RegExp(r'<think>[\s\S]*?</think>', caseSensitive: false), '');

    // Remove orphan <think> or </think> tags
    limpio = limpio.replaceAll(RegExp(r'</?think>', caseSensitive: false), '');

    // Remove leading "assistant"
    limpio = limpio.replaceFirst(RegExp(r'^\s*assistant\s*\n?', caseSensitive: false), '');
    if (limpio.startsWith('assistant')) {
      limpio = limpio.substring('assistant'.length);
    }

    // Remove ChatML tokens
    limpio = limpio.replaceAll(RegExp(r'<\|im_start\|>.*?\n?'), '');
    limpio = limpio.replaceAll('<|im_end|>', '');

    limpio = limpio.trim();
    return limpio;
  }

  Stream<String> _streamMock(String entradaUsuario) async* {
    final respuesta = 'No hay ningún modelo cargado. Ve a Ajustes para descargar o seleccionar un modelo.';
    final palabras = respuesta.split(' ');
    for (var i = 0; i < palabras.length; i++) {
      if (_cancelRequested) break;
      await Future.delayed(Duration(milliseconds: 25 + (i % 4) * 12));
      yield (i == 0 ? '' : ' ') + palabras[i];
    }
  }
}