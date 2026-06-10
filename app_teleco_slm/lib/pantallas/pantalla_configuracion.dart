/// Settings screen - translated, language selector, hidden metrics.
/// Errors only show in their corresponding section.

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:provider/provider.dart';
import '../proveedores/proveedor_app.dart';
import '../servicios/servicio_llm.dart';
import '../pantallas/pantalla_metricas.dart';
import '../utilidades/tema.dart';
import '../utilidades/traducciones.dart';
import '../widgets/badge_estado_modelo.dart';

class PantallaConfiguracion extends StatefulWidget {
  const PantallaConfiguracion({super.key});

  @override
  State<PantallaConfiguracion> createState() => _PantallaConfiguracionState();
}

class _PantallaConfiguracionState extends State<PantallaConfiguracion> {
  final TextEditingController _serverController = TextEditingController();
  bool _isDownloading = false;
  double _downloadProgress = 0.0;
  String? _downloadError;
  int _devTapCount = 0;

  // Static so it persists across screen navigations
  static bool _devModeStatic = false;
  bool _devMode = _devModeStatic;

  // Separate error tracking per section
  String? _localModelError;
  String? _remoteServerError;

  @override
  void initState() {
    super.initState();
    final provider = context.read<AppProvider>();
    _serverController.text = provider.settings.serverUrl;
  }

  @override
  void dispose() {
    _serverController.dispose();
    super.dispose();
  }

  bool get _isDesktop => Platform.isWindows || Platform.isLinux || Platform.isMacOS;

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<AppProvider>();

    return Scaffold(
      appBar: AppBar(
        title: Text(Tr.get('settings')),
        leading: IconButton(
          onPressed: () => Navigator.pop(context),
          icon: const Icon(Icons.arrow_back_rounded, size: 20),
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.all(24),
        physics: const BouncingScrollPhysics(),
        children: [
          _sectionHeader(Tr.get('localModel')),
          const SizedBox(height: 12),
          _isDesktop
              ? _buildModelCard(context, provider)
              : _buildAndroidModelCard(context, provider),
          const SizedBox(height: 32),

          _sectionHeader(Tr.get('remoteServer')),
          const SizedBox(height: 12),
          _buildServerCard(context, provider),
          const SizedBox(height: 32),

          // Hidden metrics - only show after 5 taps on About logo
          if (_devMode) ...[
            _sectionHeader(Tr.get('performance')),
            const SizedBox(height: 12),
            _box(
              child: InkWell(
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const PantallaMetricas()),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.speed_rounded, color: Tema.primario, size: 20),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(Tr.get('inferenceMetrics'),
                            style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: Tema.textoBase)),
                          const SizedBox(height: 2),
                          Text(Tr.get('metricsSubtitle'),
                            style: const TextStyle(fontSize: 12, color: Tema.textoApagado)),
                        ],
                      ),
                    ),
                    const Icon(Icons.chevron_right_rounded, color: Tema.textoApagado),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 32),
          ],

          _sectionHeader(Tr.get('data')),
          const SizedBox(height: 12),
          _buildDangerCard(context, provider),
          const SizedBox(height: 32),

          _sectionHeader(Tr.get('about')),
          const SizedBox(height: 12),
          _buildAboutCard(),
          const SizedBox(height: 40),
        ],
      ),
    );
  }

  Widget _sectionHeader(String title) {
    return Text(
      title,
      style: const TextStyle(
        fontSize: 11,
        fontWeight: FontWeight.w700,
        color: Tema.textoApagado,
        letterSpacing: 1.2,
      ),
    );
  }

  Widget _box({required Widget child}) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Tema.fondoTarjeta,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Tema.borde, width: 0.5),
      ),
      child: child,
    );
  }

  Widget _errorBox(String error) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Tema.error.withOpacity(0.08),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.only(top: 1),
            child: Icon(Icons.error_outline, color: Tema.error, size: 15),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              error,
              style: const TextStyle(color: Tema.error, fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }

  // =========================================================================
  // SERVER CARD
  // =========================================================================

  Widget _buildServerCard(BuildContext context, AppProvider provider) {
    return _box(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.cloud_rounded, color: Tema.primario, size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  Tr.get('remoteServerTitle'),
                  style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: Tema.textoBase),
                ),
              ),
              if (provider.activeSource == ActiveSource.remoteServer)
                BadgeEstadoModelo(estado: provider.modelStatus, compacto: true),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            Tr.get('serverHint'),
            style: const TextStyle(fontSize: 12, color: Tema.textoApagado),
          ),
          const SizedBox(height: 14),
          TextField(
            controller: _serverController,
            style: const TextStyle(fontSize: 14, color: Tema.textoBase),
            decoration: InputDecoration(
              hintText: 'e.g. 192.168.1.100:8089',
              hintStyle: const TextStyle(color: Tema.textoApagado, fontSize: 13),
              filled: true,
              fillColor: Tema.fondoBase,
              contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(10),
                borderSide: const BorderSide(color: Tema.borde),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(10),
                borderSide: const BorderSide(color: Tema.borde),
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(10),
                borderSide: const BorderSide(color: Tema.bordeFoco),
              ),
              prefixIcon: const Icon(Icons.link_rounded, size: 18, color: Tema.textoApagado),
            ),
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              Expanded(
                child: _actionButton(
                  label: Tr.get('connect'),
                  icon: Icons.wifi_rounded,
                  onTap: () async {
                    final url = _serverController.text.trim();
                    if (url.isNotEmpty) {
                      setState(() => _remoteServerError = null);
                      await provider.connectToServer(url);
                      if (mounted && provider.llm.errorCarga != null) {
                        setState(() {
                          _remoteServerError = provider.llm.errorCarga;
                        });
                      }
                    }
                  },
                ),
              ),
              if (provider.isModelLoaded && provider.activeSource == ActiveSource.remoteServer) ...[
                const SizedBox(width: 8),
                _actionButton(
                  label: Tr.get('disconnect'),
                  icon: Icons.wifi_off_rounded,
                  onTap: () {
                    provider.disconnectServer();
                    setState(() => _remoteServerError = null);
                  },
                  compact: true,
                  danger: true,
                ),
              ],
            ],
          ),
          // Only show server errors here
          if (_remoteServerError != null) ...[
            const SizedBox(height: 12),
            _errorBox(_remoteServerError!),
          ],
        ],
      ),
    );
  }

  // =========================================================================
  // MODEL CARD (Windows)
  // =========================================================================

  Widget _buildModelCard(BuildContext context, AppProvider provider) {
    final modelPath = provider.settings.rutaModelo;

    return _box(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.memory_rounded, color: Tema.primario, size: 20),
              const SizedBox(width: 8),
              const Expanded(
                child: Text(
                  'Qwen 3 1.7B Fine-tuned',
                  style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: Tema.textoBase),
                ),
              ),
              if (provider.activeSource == ActiveSource.localModel)
                BadgeEstadoModelo(estado: provider.modelStatus, compacto: true),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            Tr.get('formatInfo'),
            style: const TextStyle(fontSize: 12, color: Tema.textoApagado),
          ),
          const SizedBox(height: 14),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Tema.fondoBase,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: Tema.borde, width: 0.5),
            ),
            child: Row(
              children: [
                const Icon(Icons.folder_outlined, size: 14, color: Tema.textoApagado),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    modelPath.isEmpty ? Tr.get('noModel') : modelPath,
                    style: TextStyle(
                      fontSize: 12,
                      fontFamily: 'monospace',
                      color: modelPath.isEmpty ? Tema.textoApagado : Tema.textoSecundario,
                    ),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),
          Column(
            children: [
              _actionButton(
                label: 'Descargar modelo',
                icon: Icons.cloud_download_rounded,
                onTap: _isDownloading ? () {} : () => _downloadModel(context, provider),
              ),
              const SizedBox(height: 8),
              _actionButton(
                label: Tr.get('selectGguf'),
                icon: Icons.file_open_rounded,
                onTap: () => _pickModel(context, provider),
              ),
            ],
          ),
          // Only show local model errors here
          if (_localModelError != null) ...[
            const SizedBox(height: 12),
            _errorBox(_localModelError!),
          ],
        ],
      ),
    );
  }

  // =========================================================================
  // ANDROID MODEL CARD
  // =========================================================================

  Widget _buildAndroidModelCard(BuildContext context, AppProvider provider) {
    final hasLocalModel = provider.llm.localModelPath != null;
    final isLocalActive = provider.activeSource == ActiveSource.localModel;
    final isLoading = provider.modelStatus == ModelStatus.loading;
    final isError = provider.modelStatus == ModelStatus.error;

    return _box(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.memory_rounded, color: Tema.primario, size: 20),
              const SizedBox(width: 8),
              const Expanded(
                child: Text(
                  'Qwen 3 1.7B Fine-tuned',
                  style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: Tema.textoBase),
                ),
              ),
              BadgeEstadoModelo(estado: provider.modelStatus, compacto: true),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            hasLocalModel
                ? '${Tr.get('modelLoaded')} ${provider.llm.localModelPath!.split('/').last}'
                : isError
                    ? 'El modelo no se pudo cargar'
                    : Tr.get('formatInfo'),
            style: const TextStyle(fontSize: 12, color: Tema.textoApagado),
          ),

          // Show RAM info if available
          if (provider.llm.lastFreeRamMB != null) ...[
            const SizedBox(height: 4),
            Text(
              'RAM libre: ${provider.llm.lastFreeRamMB} MB',
              style: TextStyle(
                fontSize: 11,
                color: provider.llm.lastFreeRamMB! >= 1800
                    ? Colors.green[400]
                    : Tema.error,
              ),
            ),
          ],

          const SizedBox(height: 14),

          if (isLoading) ...[
            const Center(
              child: Padding(
                padding: EdgeInsets.symmetric(vertical: 8),
                child: Column(
                  children: [
                    CircularProgressIndicator(strokeWidth: 2),
                    SizedBox(height: 8),
                    Text('Cargando modelo...',
                      style: TextStyle(fontSize: 12, color: Tema.textoApagado)),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 14),
          ],

          if (_isDownloading) ...[
            Column(
              children: [
                LinearProgressIndicator(
                  value: _downloadProgress,
                  backgroundColor: Tema.fondoBase,
                  color: Tema.primario,
                ),
                const SizedBox(height: 6),
                Text('${Tr.get('downloading')} ${(_downloadProgress * 100).toStringAsFixed(1)}%',
                  style: const TextStyle(fontSize: 12, color: Tema.textoSecundario)),
              ],
            ),
            const SizedBox(height: 14),
          ],

          if (_downloadError != null) ...[
            _errorBox(_downloadError!),
            const SizedBox(height: 14),
          ],

          // Retry button when model failed to load (e.g. RAM issue)
          if (isError && !isLoading && !_isDownloading) ...[
            _actionButton(
              label: 'Reintentar carga',
              icon: Icons.refresh_rounded,
              onTap: () async {
                setState(() => _localModelError = null);
                final modelPath = await ServicioLLM.findLocalModel();
                if (modelPath != null) {
                  await provider.loadLocalModel(modelPath);
                  if (mounted && provider.llm.errorCarga != null) {
                    setState(() => _localModelError = provider.llm.errorCarga);
                  }
                }
              },
            ),
            const SizedBox(height: 8),
          ],

          Column(
            children: [
              _actionButton(
                label: Tr.get('download'),
                icon: Icons.cloud_download_rounded,
                onTap: _isDownloading ? () {} : () => _downloadModel(context, provider),
              ),
              const SizedBox(height: 8),
              _actionButton(
                label: Tr.get('selectGguf'),
                icon: Icons.folder_open_rounded,
                onTap: () => _pickAndroidModel(context, provider),
              ),
            ],
          ),
          if (isLocalActive) ...[
            const SizedBox(height: 8),
            _actionButton(
              label: Tr.get('unload'),
              icon: Icons.eject_rounded,
              onTap: provider.unloadLocalModel,
              compact: true,
              danger: true,
            ),
          ],
          // Only show local model errors here
          if (_localModelError != null) ...[
            const SizedBox(height: 12),
            _errorBox(_localModelError!),
          ],
        ],
      ),
    );
  }

  Future<void> _downloadModel(BuildContext context, AppProvider provider) async {
    setState(() {
      _isDownloading = true;
      _downloadProgress = 0.0;
      _downloadError = null;
      _localModelError = null;
    });

    String? filePath;

    try {
      final modelDir = await ServicioLLM.getModelDirectory();
      filePath = '$modelDir/qwen3-1.7b-finetuned-q4_k_m.gguf';

      if (File(filePath).existsSync()) {
        setState(() { _isDownloading = false; });
        await provider.loadLocalModel(filePath);
        if (mounted && provider.llm.errorCarga != null) {
          setState(() => _localModelError = provider.llm.errorCarga);
        }
        return;
      }

      const url = 'https://huggingface.co/arturofierrop/qwen-3-1.7B-teleco-slm-GGUF/resolve/main/qwen3-1.7b-finetuned-q4_k_m.gguf?download=true';

      final client = HttpClient();
      client.connectionTimeout = const Duration(seconds: 30);
      final request = await client.getUrl(Uri.parse(url));
      request.followRedirects = true;
      request.maxRedirects = 10;
      final response = await request.close();

      if (response.statusCode != 200) {
        throw Exception('Servidor respondió con código ${response.statusCode}');
      }

      final totalBytes = response.contentLength;
      var receivedBytes = 0;
      var lastProgressUpdate = 0.0;
      final file = File(filePath);
      final sink = file.openWrite();
      var chunksSinceFlush = 0;

      await for (final chunk in response) {
        sink.add(chunk);
        receivedBytes += chunk.length;
        chunksSinceFlush++;

        // Flush every ~5 MB to avoid memory buildup
        if (chunksSinceFlush >= 80) {
          await sink.flush();
          chunksSinceFlush = 0;
        }

        // Only update UI every 1% to avoid overwhelming setState
        if (totalBytes > 0) {
          final progress = receivedBytes / totalBytes;
          if (progress - lastProgressUpdate >= 0.01) {
            lastProgressUpdate = progress;
            if (mounted) {
              setState(() {
                _downloadProgress = progress;
              });
            }
          }
        }
      }

      await sink.flush();
      await sink.close();
      client.close();

      // Verify file size
      final downloadedSize = await file.length();
      if (totalBytes > 0 && downloadedSize < totalBytes * 0.99) {
        await file.delete();
        throw Exception('Descarga incompleta ($downloadedSize / $totalBytes bytes)');
      }

      if (mounted) {
        setState(() {
          _isDownloading = false;
        });
        // Don't auto-load right after download — RAM may be full from download buffers.
        // Let user restart app so Android frees memory first.
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Modelo descargado. Cierra y reabre la app para cargarlo.'),
            duration: Duration(seconds: 5),
          ),
        );
      }

    } catch (e) {
      // Delete partial file to prevent crash on next app open
      if (filePath != null) {
        try {
          final partial = File(filePath);
          if (await partial.exists()) await partial.delete();
        } catch (_) {}
      }

      if (mounted) {
        setState(() {
          _isDownloading = false;
          _downloadError = '${Tr.get('downloadFailed')} $e';
        });
      }
    }
  }

  Future<void> _pickAndroidModel(BuildContext context, AppProvider provider) async {
    setState(() => _localModelError = null);
    final result = await FilePicker.platform.pickFiles(
      type: FileType.any,
      dialogTitle: Tr.get('selectGguf'),
    );
    if (result != null && result.files.single.path != null) {
      final pickedPath = result.files.single.path!;
      final fileName = pickedPath.split('/').last;

      // Copy from cache to permanent models directory
      try {
        final modelDir = await ServicioLLM.getModelDirectory();
        final permanentPath = '$modelDir/$fileName';
        final permanentFile = File(permanentPath);

        if (!permanentFile.existsSync()) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('Copiando modelo al almacenamiento permanente...'),
                duration: Duration(seconds: 2),
              ),
            );
          }
          await File(pickedPath).copy(permanentPath);
        }

        await provider.loadLocalModel(permanentPath);
      } catch (e) {
        // Fallback: try loading from original path
        await provider.loadLocalModel(pickedPath);
      }

      if (mounted && provider.llm.errorCarga != null) {
        setState(() => _localModelError = provider.llm.errorCarga);
      }
    }
  }

  // =========================================================================
  // SHARED WIDGETS
  // =========================================================================

  Widget _buildDangerCard(BuildContext context, AppProvider provider) {
    return _box(
      child: _actionButton(
        label: Tr.get('clearHistory'),
        icon: Icons.delete_forever_rounded,
        onTap: () => _confirmClear(context, provider),
        danger: true,
      ),
    );
  }

  Widget _buildAboutCard() {
    return _box(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              GestureDetector(
                onTap: () {
                  _devTapCount++;
                  if (_devTapCount >= 5 && !_devMode) {
                    setState(() {
                      _devMode = true;
                      _devModeStatic = true;
                    });
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text('Métricas de inferencia activadas'),
                        duration: Duration(seconds: 2),
                      ),
                    );
                  }
                },
                child: Container(
                  width: 32, height: 32,
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(colors: [Tema.primario, Tema.acento]),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(6),
                    child: Image.asset('assets/iconos/app_logo.png', width: 28, height: 28),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              const Text('Teleco SLM v1.0.0',
                style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: Tema.textoBase)),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            _isDesktop ? Tr.get('aboutDesktop') : Tr.get('aboutMobile'),
            style: const TextStyle(fontSize: 13, color: Tema.textoApagado, height: 1.55),
          ),
          const SizedBox(height: 14),
          const Divider(),
          const SizedBox(height: 10),
          Row(
            children: [
              const _InfoChip(icon: Icons.memory_rounded, text: 'llama.cpp'),
              const SizedBox(width: 10),
              const _InfoChip(icon: Icons.storage_rounded, text: 'SQLite'),
              const SizedBox(width: 10),
              _InfoChip(
                icon: _isDesktop ? Icons.wifi_off_rounded : Icons.wifi_rounded,
                text: _isDesktop ? 'Offline' : 'Network',
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _actionButton({
    required String label,
    required IconData icon,
    required VoidCallback onTap,
    bool compact = false,
    bool danger = false,
  }) {
    final color = danger ? Tema.error : Tema.textoSecundario;
    final bgColor = danger ? Tema.error : Tema.primario;

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(10),
        child: Container(
          padding: EdgeInsets.symmetric(
            horizontal: compact ? 12 : 14, vertical: 11),
          decoration: BoxDecoration(
            color: bgColor.withOpacity(0.07),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: bgColor.withOpacity(0.12), width: 0.5),
          ),
          child: Row(
            mainAxisSize: compact ? MainAxisSize.min : MainAxisSize.max,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, size: 16, color: color),
              const SizedBox(width: 6),
              Text(label, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w500, color: color)),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _pickModel(BuildContext context, AppProvider provider) async {
    setState(() => _localModelError = null);
    final result = await FilePicker.platform.pickFiles(
      type: FileType.any,
      dialogTitle: Tr.get('selectGguf'),
    );
    if (result != null && result.files.single.path != null) {
      await provider.loadModel(result.files.single.path!);
      if (mounted && provider.llm.errorCarga != null) {
        setState(() => _localModelError = provider.llm.errorCarga);
      }
    }
  }

  Future<void> _confirmClear(BuildContext context, AppProvider provider) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(Tr.get('clearConfirmTitle')),
        content: Text(Tr.get('clearConfirmBody')),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: Text(Tr.get('cancel'))),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: TextButton.styleFrom(foregroundColor: Tema.error),
            child: Text(Tr.get('clearAll')),
          ),
        ],
      ),
    );
    if (confirmed == true) {
      await provider.clearAllHistory();
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(Tr.get('historyCleared'))),
        );
      }
    }
  }
}

class _InfoChip extends StatelessWidget {
  final IconData icon;
  final String text;
  const _InfoChip({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: Tema.fondoBase,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: Tema.borde, width: 0.5),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 12, color: Tema.textoApagado),
          const SizedBox(width: 4),
          Text(text, style: const TextStyle(fontSize: 11, color: Tema.textoApagado)),
        ],
      ),
    );
  }
}