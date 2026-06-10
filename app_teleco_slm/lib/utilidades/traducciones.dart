/// Internationalization - Spanish (default) and English.
library;

class Tr {
  Tr._();

  static String _lang = 'es';

  static String get lang => _lang;
  static bool get isEs => _lang == 'es';
  static bool get isEn => _lang == 'en';

  static void setLang(String l) => _lang = l;

  static final Map<String, Map<String, String>> _t = {
    // === General ===
    'appName': {'es': 'Teleco SLM', 'en': 'Teleco SLM'},
    'settings': {'es': 'Ajustes', 'en': 'Settings'},

    // === Home ===
    'newChat': {'es': 'Nuevo Chat', 'en': 'New Chat'},
    'startConversation': {'es': 'Iniciar una nueva conversación', 'en': 'Start a new conversation'},
    'recentChats': {'es': 'CHATS RECIENTES', 'en': 'RECENT CHATS'},

    // === Chat ===
    'askQuestion': {'es': 'Haz una pregunta', 'en': 'Ask a question'},
    'localInference': {'es': 'Inferencia local en CPU - 100% sin conexión', 'en': 'Local CPU inference - 100% offline'},
    'typeMessage': {'es': 'Escribe tu mensaje...', 'en': 'Type your message...'},
    'waitForModel': {'es': 'Espera a que el modelo termine de responder', 'en': 'Wait for the model to finish responding'},

    // === Settings sections ===
    'localModel': {'es': 'MODELO LOCAL', 'en': 'LOCAL MODEL'},
    'remoteServer': {'es': 'SERVIDOR REMOTO', 'en': 'REMOTE SERVER'},
    'performance': {'es': 'RENDIMIENTO', 'en': 'PERFORMANCE'},
    'data': {'es': 'DATOS', 'en': 'DATA'},
    'about': {'es': 'ACERCA DE', 'en': 'ABOUT'},
    'language': {'es': 'IDIOMA', 'en': 'LANGUAGE'},

    // === Server card ===
    'remoteServerTitle': {'es': 'Servidor Remoto', 'en': 'Remote Server'},
    'serverHint': {'es': 'Conectar a un llama-server en tu red', 'en': 'Connect to a llama-server running on your network'},
    'connect': {'es': 'Conectar', 'en': 'Connect'},
    'disconnect': {'es': 'Desconectar', 'en': 'Disconnect'},

    // === Model card ===
    'localModelTitle': {'es': 'Modelo Local', 'en': 'Local Model'},
    'noModel': {'es': 'Sin modelo seleccionado', 'en': 'No model selected'},
    'noLocalModel': {'es': 'Sin modelo local. Descarga o selecciona un archivo .gguf.', 'en': 'No local model. Download or select a .gguf file.'},
    'modelLoaded': {'es': 'Modelo cargado:', 'en': 'Model loaded:'},
    'selectGguf': {'es': 'Seleccionar archivo .gguf', 'en': 'Select .gguf file'},
    'unload': {'es': 'Descargar', 'en': 'Unload'},
    'download': {'es': 'Descargar modelo', 'en': 'Download'},
    'downloading': {'es': 'Descargando...', 'en': 'Downloading...'},
    'downloadFailed': {'es': 'Error en la descarga:', 'en': 'Download failed:'},
    'formatInfo': {'es': 'Formato Q4_K_M · ~1.1 GB · Solo CPU', 'en': 'Q4_K_M format · ~1.1 GB · CPU only'},

    // === Metrics ===
    'inferenceMetrics': {'es': 'Métricas de Inferencia', 'en': 'Inference Metrics'},
    'metricsSubtitle': {'es': 'Tokens/s, latencia, tiempo por petición', 'en': 'Tokens/s, latency, time per request'},
    'noMetrics': {'es': 'Sin métricas aún.\nEnvía un mensaje para empezar a medir.', 'en': 'No metrics yet.\nSend a message to start measuring.'},
    'tokensGenerated': {'es': 'Tokens generados', 'en': 'Tokens generated'},
    'timeToFirstToken': {'es': 'Tiempo al primer token', 'en': 'Time to first token'},
    'totalTime': {'es': 'Tiempo total', 'en': 'Total time'},
    'speed': {'es': 'Velocidad', 'en': 'Speed'},
    'memory': {'es': 'Memoria (RSS)', 'en': 'Memory (RSS)'},

    // === Data ===
    'clearHistory': {'es': 'Borrar todo el historial', 'en': 'Clear all chat history'},
    'clearConfirmTitle': {'es': '¿Borrar historial?', 'en': 'Clear history?'},
    'clearConfirmBody': {'es': 'Todas las conversaciones y mensajes se eliminarán. No se puede deshacer.', 'en': 'All conversations and messages will be deleted. This cannot be undone.'},
    'cancel': {'es': 'Cancelar', 'en': 'Cancel'},
    'clearAll': {'es': 'Borrar todo', 'en': 'Clear all'},
    'historyCleared': {'es': 'Historial borrado', 'en': 'History cleared'},

    // === About ===
    'aboutDesktop': {'es': 'Inferencia LLM local para ingeniería de telecomunicaciones. Ejecuta Qwen 3 1.7B Fine-tuned con llama.cpp en CPU, completamente offline.', 'en': 'Local LLM inference for telecommunications engineering. Runs Qwen 3 1.7B Fine-tuned with llama.cpp on CPU, fully offline.'},
    'aboutMobile': {'es': 'Inferencia LLM local para ingeniería de telecomunicaciones. Ejecuta Qwen 3 1.7B Fine-tuned con llama.cpp en CPU, completamente offline.', 'en': 'Local LLM inference for telecommunications engineering. Runs Qwen 3 1.7B Fine-tuned with llama.cpp on CPU, fully offline.'},

    // === Language ===
    'spanish': {'es': 'Español', 'en': 'Spanish'},
    'english': {'es': 'Inglés', 'en': 'English'},

    // === Splash ===
    'splashSubtitle': {'es': 'Inferencia local · Sin conexión · CPU', 'en': 'Local inference · Offline · CPU'},

    // === Suggestions ===
    'sug1': {'es': '¿Qué es OFDM y cómo funciona?', 'en': 'What is OFDM and how does it work?'},
    'sug2': {'es': 'Explica el teorema de Shannon', 'en': 'Explain Shannon\'s theorem'},
    'sug3': {'es': '¿Cómo funciona MIMO masivo en 5G?', 'en': 'How does massive MIMO work in 5G?'},
    'sug4': {'es': 'Explica la modulación QAM paso a paso', 'en': 'Explain QAM modulation step by step'},
    'sug5': {'es': '¿Qué es la transformada de Fourier?', 'en': 'What is the Fourier transform?'},
  };

  static String get(String key) {
    return _t[key]?[_lang] ?? _t[key]?['es'] ?? key;
  }

  static List<String> get suggestions => [
    get('sug1'), get('sug2'), get('sug3'), get('sug4'), get('sug5'),
  ];
}