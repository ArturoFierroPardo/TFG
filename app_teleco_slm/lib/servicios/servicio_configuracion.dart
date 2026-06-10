/// Settings persistence service.
library;

import 'package:shared_preferences/shared_preferences.dart';

class ServicioConfiguracion {
  static const _keyModelPath = 'ruta_modelo';
  static const _keyServerUrl = 'server_url';
  static const _keyIdioma = 'idioma';

  late SharedPreferences _prefs;

  Future<void> inicializar() async {
    _prefs = await SharedPreferences.getInstance();
  }

  // Model path (Windows)
  String get rutaModelo => _prefs.getString(_keyModelPath) ?? '';
  Future<void> guardarRutaModelo(String path) =>
      _prefs.setString(_keyModelPath, path);

  // Server URL (Android)
  String get serverUrl => _prefs.getString(_keyServerUrl) ?? '';
  Future<void> guardarServerUrl(String url) =>
      _prefs.setString(_keyServerUrl, url);

  // Language
  String get idioma => _prefs.getString(_keyIdioma) ?? 'es';
  Future<void> guardarIdioma(String lang) =>
      _prefs.setString(_keyIdioma, lang);
}
