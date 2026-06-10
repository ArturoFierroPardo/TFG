/// Burbuja de chat con Markdown, LaTeX y texto seleccionable.
///
/// - Limpia tags de thinking del modelo automáticamente.
/// - Convierte $...$ a \(...\) para gpt_markdown.
/// - Auto-envuelve LaTeX sin delimitadores (safety net con try-catch).
/// - Texto seleccionable en todo el contenido.
/// - Solo muestra icono de rol.
library;

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:gpt_markdown/gpt_markdown.dart';
import '../servicios/servicio_llm.dart';
import '../utilidades/tema.dart';

class BurbujaChat extends StatelessWidget {
  final String contenido;
  final bool esUsuario;
  final bool enStreaming;
  final DateTime? marca;

  const BurbujaChat({
    super.key,
    required this.contenido,
    required this.esUsuario,
    this.enStreaming = false,
    this.marca,
  });

  @override
  Widget build(BuildContext context) {
    final textoLimpio = esUsuario
        ? contenido
        : _prepararTextoAsistente(contenido);

    return Padding(
      padding: EdgeInsets.only(
        left: esUsuario ? 52 : 0,
        right: esUsuario ? 0 : 52,
        bottom: 14,
      ),
      child: Align(
        alignment: esUsuario ? Alignment.centerRight : Alignment.centerLeft,
        child: Column(
          crossAxisAlignment:
              esUsuario ? CrossAxisAlignment.end : CrossAxisAlignment.start,
          children: [
            // Solo icono, sin texto
            Padding(
              padding: const EdgeInsets.only(bottom: 5, left: 4, right: 4),
              child: Icon(
                esUsuario ? Icons.person_rounded : Icons.smart_toy_rounded,
                size: 14,
                color: Tema.textoApagado,
              ),
            ),

            // Burbuja
            GestureDetector(
              onLongPress: () {
                if (textoLimpio.isNotEmpty) {
                  Clipboard.setData(ClipboardData(text: textoLimpio));
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('Copiado al portapapeles'),
                      duration: Duration(seconds: 1),
                    ),
                  );
                }
              },
              child: AnimatedContainer(
                duration: Tema.duracionNormal,
                curve: Tema.curvaEntrada,
                decoration: BoxDecoration(
                  color: esUsuario
                      ? Tema.burbujaUsuario
                      : Tema.burbujaAsistente,
                  borderRadius: BorderRadius.only(
                    topLeft: const Radius.circular(18),
                    topRight: const Radius.circular(18),
                    bottomLeft: Radius.circular(esUsuario ? 18 : 4),
                    bottomRight: Radius.circular(esUsuario ? 4 : 18),
                  ),
                  border: esUsuario
                      ? null
                      : Border.all(color: Tema.borde, width: 0.5),
                  boxShadow: esUsuario
                      ? [
                          BoxShadow(
                            color: Tema.primario.withOpacity(0.15),
                            blurRadius: 12,
                            offset: const Offset(0, 4),
                          ),
                        ]
                      : null,
                ),
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                child: textoLimpio.isEmpty && enStreaming
                    ? const _PuntosEscribiendo()
                    : _construirContenido(context, textoLimpio),
              ),
            ),

            // Hint de tiempo cuando el modelo está pensando (solo Android)
            if (!esUsuario && enStreaming && textoLimpio.isEmpty && Platform.isAndroid)
              Padding(
                padding: const EdgeInsets.only(top: 6, left: 4),
                child: const Text(
                  'El modelo local puede tardar 10-60s según el dispositivo',
                  style: TextStyle(
                    fontSize: 10.5,
                    color: Tema.textoApagado,
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  // Spanish short words — must NOT be treated as math variables
  static const _spanishShort = {
    'y', 'la', 'el', 'en', 'de', 'un', 'se', 'es', 'al', 'no',
    'si', 'lo', 'su', 'ya', 'o', 'a', 'e', 'u', 'ni', 'me', 'te',
  };

  /// Limpia y prepara el texto del asistente.
  String _prepararTextoAsistente(String texto) {
    var limpio = ServicioLLM.limpiarSalida(texto);
    if (!enStreaming) {
      limpio = _autoEnvolverLatex(limpio);
      limpio = _asegurarEspacioFormulas(limpio);
      limpio = _convertirDelimitadoresLatex(limpio);
    }
    return limpio;
  }

  /// Ensures block formulas ($$...$$) have line breaks before and after.
  static String _asegurarEspacioFormulas(String texto) {
    // Add newline before $$ if there's text immediately before
    var result = texto.replaceAllMapped(
      RegExp(r'([^\n])\$\$'),
      (m) => '${m.group(1)}\n\$\$',
    );
    // Add newline after $$ closing if there's text immediately after
    result = result.replaceAllMapped(
      RegExp(r'\$\$([^\n\$])'),
      (m) => '\$\$\n${m.group(1)}',
    );
    return result;
  }

  /// Word-level auto-wrapper: splits text by spaces, identifies words
  /// containing \commands, groups consecutive math tokens, wraps in $..$.
  /// Never eats Spanish words because it checks each word individually.
  static String _autoEnvolverLatex(String texto) {
    try {
      if (texto.contains(r'\(') || texto.contains(r'\[')) return texto;
      if (RegExp(r'\$[^$]+\$').hasMatch(texto)) return texto;
      if (!RegExp(r'\\[a-zA-Z]').hasMatch(texto)) return texto;

      final words = texto.split(' ');
      final result = <String>[];
      final mathBuffer = <String>[];

      void flushMath() {
        if (mathBuffer.isNotEmpty) {
          result.add('\$${mathBuffer.join(' ')}\$');
          mathBuffer.clear();
        }
      }

      for (final word in words) {
        // Separate trailing punctuation
        var stripped = word;
        var trailing = '';
        while (stripped.isNotEmpty && '.,;:!?'.contains(stripped[stripped.length - 1])) {
          trailing = stripped[stripped.length - 1] + trailing;
          stripped = stripped.substring(0, stripped.length - 1);
        }

        bool isMath = false;

        // Word contains a \command → definitely math
        if (stripped.contains(r'\') && RegExp(r'\\[a-zA-Z]').hasMatch(stripped)) {
          isMath = true;
        }
        // If we're in a math sequence, check if this word continues it
        else if (mathBuffer.isNotEmpty && _isMathToken(stripped)) {
          isMath = true;
        }

        if (isMath) {
          if (trailing.isNotEmpty) {
            mathBuffer.add(stripped);
            flushMath();
            result[result.length - 1] = result[result.length - 1] + trailing;
          } else {
            mathBuffer.add(stripped);
          }
        } else {
          flushMath();
          result.add(word);
        }
      }

      flushMath();
      return result.join(' ');
    } catch (_) {
      return texto;
    }
  }

  /// Check if a token is part of a math expression (not a Spanish word).
  static bool _isMathToken(String token) {
    if (token.isEmpty) return false;
    if (token.contains(r'\')) return true;
    final low = token.toLowerCase();
    if (_spanishShort.contains(low)) return false;
    // Short math variable/number (x, R, dx, 3, etc.)
    if (token.length <= 3 && RegExp(r'^[A-Za-z0-9_{}^()\[\]+\-*/=]+$').hasMatch(token)) {
      return true;
    }
    // Contains braces/sub/superscripts → math
    if (RegExp(r'^[A-Za-z0-9_{}^()\[\]+\-*/=.,]+$').hasMatch(token) &&
        (token.contains('{') || token.contains('^') || token.contains('_'))) {
      return true;
    }
    // Single operator
    if (const ['+', '-', '*', '/', '=', '<', '>', '|'].contains(token)) {
      return true;
    }
    return false;
  }

  /// Converts $...$ → \(...\) and $$...$$ → \[...\] so gpt_markdown renders them.
  static String _convertirDelimitadoresLatex(String texto) {
    try {
      // First convert block math: $$...$$ → \[...\]
      var result = texto.replaceAllMapped(
        RegExp(r'\$\$([\s\S]*?)\$\$'),
        (m) => '\\[${m.group(1)}\\]',
      );
      // Then convert inline math: $...$ → \(...\)
      result = result.replaceAllMapped(
        RegExp(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)'),
        (m) => '\\(${m.group(1)}\\)',
      );
      return result;
    } catch (_) {
      return texto; // On any error, return original
    }
  }

  // =========================================================================
  // CONTENT BUILDER
  // =========================================================================

  Widget _construirContenido(BuildContext context, String texto) {
    if (esUsuario) {
      return SelectableText(
        texto,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 14.5,
          height: 1.45,
        ),
      );
    }

    // Asistente: Markdown + LaTeX, seleccionable
    // SingleChildScrollView handles formulas wider than screen
    return SelectionArea(
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        physics: const ClampingScrollPhysics(),
        child: ConstrainedBox(
          constraints: BoxConstraints(
            minWidth: 0,
            maxWidth: MediaQuery.of(context).size.width - (esUsuario ? 52 : 52) - 32 - 16,
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              GptMarkdown(
                texto,
                style: const TextStyle(
                  color: Tema.textoBase,
                  fontSize: 14.5,
                  height: 1.55,
                ),
              ),
              if (enStreaming) ...[
                const SizedBox(height: 4),
                const _CursorParpadeante(),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Cursor parpadeante (durante streaming)
// ---------------------------------------------------------------------------
class _CursorParpadeante extends StatefulWidget {
  const _CursorParpadeante();

  @override
  State<_CursorParpadeante> createState() => _CursorParpadeanteState();
}

class _CursorParpadeanteState extends State<_CursorParpadeante>
    with SingleTickerProviderStateMixin {
  late AnimationController _controlador;

  @override
  void initState() {
    super.initState();
    _controlador = AnimationController(
      duration: const Duration(milliseconds: 500),
      vsync: this,
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _controlador.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _controlador,
      child: Container(
        width: 2,
        height: 16,
        decoration: BoxDecoration(
          color: Tema.primarioClaro,
          borderRadius: BorderRadius.circular(1),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Puntos de "escribiendo" con rebote vertical
// ---------------------------------------------------------------------------
class _PuntosEscribiendo extends StatefulWidget {
  const _PuntosEscribiendo();

  @override
  State<_PuntosEscribiendo> createState() => _PuntosEscribiendoState();
}

class _PuntosEscribiendoState extends State<_PuntosEscribiendo>
    with TickerProviderStateMixin {
  late final List<AnimationController> _controladores;

  @override
  void initState() {
    super.initState();
    _controladores = List.generate(3, (i) {
      return AnimationController(
        duration: const Duration(milliseconds: 600),
        vsync: this,
      );
    });

    // Stagger start for wave effect
    for (var i = 0; i < 3; i++) {
      Future.delayed(Duration(milliseconds: i * 180), () {
        if (mounted) _controladores[i].repeat(reverse: true);
      });
    }
  }

  @override
  void dispose() {
    for (final c in _controladores) {
      c.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: List.generate(3, (i) {
        return AnimatedBuilder(
          animation: _controladores[i],
          builder: (_, __) {
            final bounce = -6.0 * _controladores[i].value;
            return Transform.translate(
              offset: Offset(0, bounce),
              child: Container(
                margin: const EdgeInsets.symmetric(horizontal: 3),
                width: 7,
                height: 7,
                decoration: BoxDecoration(
                  color: Tema.textoApagado
                      .withOpacity(0.4 + _controladores[i].value * 0.6),
                  shape: BoxShape.circle,
                ),
              ),
            );
          },
        );
      }),
    );
  }
}