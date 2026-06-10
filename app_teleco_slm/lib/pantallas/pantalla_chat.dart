/// Chat screen with streaming and Enter/Shift+Enter support.
library;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import '../proveedores/proveedor_app.dart';
import '../utilidades/constantes.dart';
import '../utilidades/tema.dart';
import '../utilidades/traducciones.dart';
import '../widgets/burbuja_chat.dart';
import '../widgets/badge_estado_modelo.dart';

class PantallaChat extends StatefulWidget {
  const PantallaChat({super.key});

  @override
  State<PantallaChat> createState() => _PantallaChatState();
}

class _PantallaChatState extends State<PantallaChat> {
  final TextEditingController _textController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final FocusNode _focusNode = FocusNode();

  @override
  void dispose() {
    _textController.dispose();
    _scrollController.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: Tema.duracionNormal,
          curve: Tema.curvaEntrada,
        );
      }
    });
  }

  void _sendMessage(AppProvider provider) {
    final text = _textController.text.trim();
    if (text.isEmpty) return;
    _textController.clear();
    provider.sendMessage(text);
    _scrollToBottom();
    _focusNode.requestFocus();
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<AppProvider>();
    final conv = provider.currentConversation;
    if (conv == null) return const SizedBox.shrink();

    if (provider.isCurrentChatGenerating) _scrollToBottom();

    return Scaffold(
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(58),
        child: AppBar(
          leading: IconButton(
            onPressed: () => Navigator.pop(context),
            icon: const Icon(Icons.arrow_back_rounded, size: 20),
          ),
          title: const Text('Teleco SLM', style: TextStyle(fontSize: 15)),
          actions: [
            Padding(
              padding: const EdgeInsets.only(right: 12),
              child: BadgeEstadoModelo(
                estado: provider.modelStatus,
                compacto: true,
              ),
            ),
          ],
        ),
      ),
      body: Column(
        children: [
          Expanded(
            child: provider.messages.isEmpty
                ? _buildEmptyState(provider)
                : ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
                    physics: const BouncingScrollPhysics(),
                    itemCount: provider.messages.length,
                    itemBuilder: (context, index) {
                      final msg = provider.messages[index];
                      return BurbujaChat(
                        contenido: msg.content,
                        esUsuario: msg.role == 'user',
                        enStreaming: msg.isStreaming,
                      );
                    },
                  ),
          ),
          _buildInputBar(provider),
        ],
      ),
    );
  }

  Widget _buildEmptyState(AppProvider provider) {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 68,
              height: 68,
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Tema.primario, Tema.acento],
                ),
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(
                    color: Tema.primario.withOpacity(0.25),
                    blurRadius: 20,
                    offset: const Offset(0, 6),
                  ),
                ],
              ),
              child: const Icon(Icons.chat_rounded, color: Colors.white, size: 28),
            ),
            const SizedBox(height: 22),
            Text(
              Tr.get('askQuestion'),
              style: const TextStyle(
                fontSize: 19,
                fontWeight: FontWeight.w600,
                color: Tema.textoBase,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              Tr.get('localInference'),
              style: const TextStyle(fontSize: 13, color: Tema.textoApagado),
            ),
            const SizedBox(height: 28),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              alignment: WrapAlignment.center,
              children: Tr.suggestions.map((s) {
                return Material(
                  color: Colors.transparent,
                  child: InkWell(
                    onTap: () {
                      _textController.text = s;
                      _sendMessage(provider);
                    },
                    borderRadius: BorderRadius.circular(22),
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 14, vertical: 9),
                      decoration: BoxDecoration(
                        color: Tema.fondoTarjeta,
                        borderRadius: BorderRadius.circular(22),
                        border: Border.all(color: Tema.borde, width: 0.5),
                      ),
                      child: Text(
                        s,
                        style: const TextStyle(
                          fontSize: 13,
                          color: Tema.textoSecundario,
                        ),
                      ),
                    ),
                  ),
                );
              }).toList(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInputBar(AppProvider provider) {
    return Container(
      padding: EdgeInsets.fromLTRB(
        14, 0, 14,
        MediaQuery.of(context).padding.bottom + 10,
      ),
      decoration: const BoxDecoration(
        color: Tema.fondoBase,
        border: Border(top: BorderSide(color: Tema.borde, width: 0.5)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Padding(
            padding: EdgeInsets.only(top: 8, bottom: 6),
            child: Text(
              'TelecoSLM utiliza inteligencia artificial generativa. Las respuestas pueden contener errores, datos desactualizados o información incompleta. No sustituye la consulta de fuentes académicas oficiales ni el criterio de un profesional. Por favor, verifica siempre las respuestas.',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 10.5,
                color: Tema.textoApagado,
                height: 1.3,
              ),
            ),
          ),
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Expanded(
                child: Container(
                  constraints: const BoxConstraints(maxHeight: 120),
                  child: Focus(
                    onKeyEvent: (node, event) {
                      if (event is KeyDownEvent &&
                          event.logicalKey == LogicalKeyboardKey.enter &&
                          !HardwareKeyboard.instance.isShiftPressed) {
                        _sendMessage(provider);
                        return KeyEventResult.handled;
                      }
                      return KeyEventResult.ignored;
                    },
                child: TextField(
                  controller: _textController,
                  focusNode: _focusNode,
                  maxLines: null,
                  textInputAction: TextInputAction.newline,
                  style: const TextStyle(
                    fontSize: 14.5,
                    color: Tema.textoBase,
                  ),
                  decoration: InputDecoration(
                    hintText: Tr.get('typeMessage'),
                    hintStyle: const TextStyle(color: Tema.textoApagado),
                    filled: true,
                    fillColor: Tema.fondoInput,
                    contentPadding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 12),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(24),
                      borderSide: const BorderSide(color: Tema.borde),
                    ),
                    enabledBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(24),
                      borderSide: const BorderSide(color: Tema.borde),
                    ),
                    focusedBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(24),
                      borderSide: const BorderSide(
                          color: Tema.bordeFoco, width: 1),
                    ),
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(width: 8),
          AnimatedSwitcher(
            duration: Tema.duracionRapida,
            transitionBuilder: (child, anim) =>
                ScaleTransition(scale: anim, child: child),
            child: provider.isCurrentChatGenerating
                ? _buildStopButton(provider)
                : _buildSendButton(provider),
          ),
        ],
      ),
        ],
      ),
    );
  }

  Widget _buildSendButton(AppProvider provider) {
    return SizedBox(
      key: const ValueKey('send'),
      width: 44,
      height: 44,
      child: Material(
        color: Tema.primario,
        borderRadius: BorderRadius.circular(22),
        child: InkWell(
          onTap: () {
            // If generating in another chat, show message
            if (provider.isGenerating && !provider.isCurrentChatGenerating) {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text(Tr.get('waitForModel')),
                  duration: const Duration(seconds: 2),
                ),
              );
              return;
            }
            _sendMessage(provider);
          },
          borderRadius: BorderRadius.circular(22),
          child: const Icon(Icons.arrow_upward_rounded, color: Colors.white, size: 22),
        ),
      ),
    );
  }

  Widget _buildStopButton(AppProvider provider) {
    return SizedBox(
      key: const ValueKey('stop'),
      width: 44,
      height: 44,
      child: Material(
        color: Tema.error,
        borderRadius: BorderRadius.circular(22),
        child: InkWell(
          onTap: provider.stopGeneration,
          borderRadius: BorderRadius.circular(22),
          child: const Icon(Icons.stop_rounded, color: Colors.white, size: 22),
        ),
      ),
    );
  }
}