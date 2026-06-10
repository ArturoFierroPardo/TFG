/// Tema visual de la aplicación.
///
/// Paleta oscura con acentos azul tech. Sigue la filosofía de Emil Kowalski:
/// cada detalle invisible compone la experiencia total.
library;

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class Tema {
  Tema._();

  // ═══════════════════════════════════════════════════════════════════════════
  // PALETA DE COLORES
  // ═══════════════════════════════════════════════════════════════════════════

  // --- Primarios ---
  static const Color primario = Color(0xFF2563EB);
  static const Color primarioOscuro = Color(0xFF1D4ED8);
  static const Color primarioClaro = Color(0xFF60A5FA);

  // --- Acentos ---
  static const Color acento = Color(0xFF06D6A0);
  static const Color acentoSuave = Color(0xFF00B894);

  // --- Superficies ---
  static const Color fondoBase = Color(0xFF080C14);
  static const Color fondoTarjeta = Color(0xFF0F1726);
  static const Color fondoTarjetaHover = Color(0xFF162032);
  static const Color fondoSuperficie = Color(0xFF0C1220);
  static const Color fondoInput = Color(0xFF1A2540);

  // --- Texto ---
  static const Color textoBase = Color(0xFFF0F4F8);
  static const Color textoSecundario = Color(0xFF8B9DC3);
  static const Color textoApagado = Color(0xFF5A6B8A);

  // --- Bordes ---
  static const Color borde = Color(0xFF1E2D4A);
  static const Color bordeFoco = Color(0xFF3B82F6);

  // --- Estado ---
  static const Color exito = Color(0xFF22C55E);
  static const Color advertencia = Color(0xFFF59E0B);
  static const Color error = Color(0xFFEF4444);

  // --- Burbujas de chat ---
  static const Color burbujaUsuario = Color(0xFF2563EB);
  static const Color burbujaAsistente = Color(0xFF162032);

  // ═══════════════════════════════════════════════════════════════════════════
  // CURVAS DE ANIMACIÓN (filosofía Emil: ease-out para UI, nunca ease-in)
  // ═══════════════════════════════════════════════════════════════════════════

  /// Curva principal para elementos que entran. Respuesta inmediata.
  static const Curve curvaEntrada = Cubic(0.23, 1, 0.32, 1);

  /// Curva para movimiento continuo en pantalla.
  static const Curve curvaMovimiento = Cubic(0.77, 0, 0.175, 1);

  // ═══════════════════════════════════════════════════════════════════════════
  // DURACIONES (Emil: UI < 300ms, nunca más lento)
  // ═══════════════════════════════════════════════════════════════════════════

  static const Duration duracionRapida = Duration(milliseconds: 120);
  static const Duration duracionNormal = Duration(milliseconds: 200);
  static const Duration duracionLenta = Duration(milliseconds: 300);

  // ═══════════════════════════════════════════════════════════════════════════
  // TEMA COMPLETO
  // ═══════════════════════════════════════════════════════════════════════════

  static ThemeData get temaOscuro {
    final textoBase_ = GoogleFonts.plusJakartaSansTextTheme(
      ThemeData.dark().textTheme,
    );

    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: fondoBase,

      // --- Esquema de color ---
      colorScheme: const ColorScheme.dark(
        primary: primario,
        secondary: acento,
        surface: fondoSuperficie,
        error: error,
        onPrimary: Colors.white,
        onSecondary: Colors.white,
        onSurface: textoBase,
        onError: Colors.white,
        outline: borde,
      ),

      // --- Tipografía: Plus Jakarta Sans ---
      textTheme: textoBase_.apply(
        bodyColor: textoBase,
        displayColor: textoBase,
      ),

      // --- AppBar ---
      appBarTheme: AppBarTheme(
        backgroundColor: fondoBase,
        elevation: 0,
        scrolledUnderElevation: 0,
        centerTitle: false,
        titleTextStyle: GoogleFonts.plusJakartaSans(
          fontSize: 17,
          fontWeight: FontWeight.w600,
          color: textoBase,
          letterSpacing: -0.2,
        ),
        iconTheme: const IconThemeData(color: textoBase, size: 22),
      ),

      // --- Tarjetas ---
      cardTheme: CardThemeData(
        color: fondoTarjeta,
        elevation: 0,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: const BorderSide(color: borde, width: 0.5),
        ),
      ),

      // --- Campos de texto ---
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: fondoInput,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: borde),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: borde),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: bordeFoco, width: 1.5),
        ),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        hintStyle: GoogleFonts.plusJakartaSans(
          color: textoApagado,
          fontSize: 14,
        ),
      ),

      // --- Botones elevados ---
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primario,
          foregroundColor: Colors.white,
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          textStyle: GoogleFonts.plusJakartaSans(
            fontSize: 14,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),

      // --- Sliders ---
      sliderTheme: SliderThemeData(
        activeTrackColor: primario,
        inactiveTrackColor: borde,
        thumbColor: primario,
        overlayColor: primario.withOpacity(0.12),
        trackHeight: 4,
        thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 7),
      ),

      // --- Divisores ---
      dividerTheme: const DividerThemeData(
        color: borde,
        thickness: 0.5,
        space: 0,
      ),

      // --- SnackBars ---
      snackBarTheme: SnackBarThemeData(
        backgroundColor: fondoTarjeta,
        contentTextStyle: GoogleFonts.plusJakartaSans(color: textoBase),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        behavior: SnackBarBehavior.floating,
        elevation: 8,
      ),

      // --- Dialogos ---
      dialogTheme: DialogThemeData(
        backgroundColor: fondoTarjeta,
        elevation: 16,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        titleTextStyle: GoogleFonts.plusJakartaSans(
          fontSize: 18,
          fontWeight: FontWeight.w600,
          color: textoBase,
        ),
        contentTextStyle: GoogleFonts.plusJakartaSans(
          fontSize: 14,
          color: textoSecundario,
          height: 1.5,
        ),
      ),
    );
  }
}
