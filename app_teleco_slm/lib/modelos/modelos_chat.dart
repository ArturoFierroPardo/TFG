/// Data models for the chat system.
library;

import 'package:uuid/uuid.dart';

const _uuid = Uuid();

class ChatMessage {
  final String id;
  final String conversationId;
  final String role; // 'user', 'assistant', 'system'
  final String content;
  final DateTime timestamp;
  final bool isStreaming;

  ChatMessage({
    String? id,
    required this.conversationId,
    required this.role,
    required this.content,
    DateTime? timestamp,
    this.isStreaming = false,
  })  : id = id ?? _uuid.v4(),
        timestamp = timestamp ?? DateTime.now();

  ChatMessage copyWith({String? content, bool? isStreaming}) {
    return ChatMessage(
      id: id,
      conversationId: conversationId,
      role: role,
      content: content ?? this.content,
      timestamp: timestamp,
      isStreaming: isStreaming ?? this.isStreaming,
    );
  }

  Map<String, dynamic> toMap() => {
        'id': id,
        'conversacion_id': conversationId,
        'rol': role,
        'contenido': content,
        'marca': timestamp.toIso8601String(),
      };

  factory ChatMessage.fromMap(Map<String, dynamic> map) => ChatMessage(
        id: map['id'] as String,
        conversationId: map['conversacion_id'] as String,
        role: map['rol'] as String,
        content: map['contenido'] as String,
        timestamp: DateTime.parse(map['marca'] as String),
      );
}

class Conversation {
  final String id;
  final String title;
  final DateTime createdAt;
  final DateTime updatedAt;

  Conversation({
    String? id,
    required this.title,
    DateTime? createdAt,
    DateTime? updatedAt,
  })  : id = id ?? _uuid.v4(),
        createdAt = createdAt ?? DateTime.now(),
        updatedAt = updatedAt ?? DateTime.now();

  Conversation copyWith({String? title, DateTime? updatedAt}) {
    return Conversation(
      id: id,
      title: title ?? this.title,
      createdAt: createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
    );
  }

  Map<String, dynamic> toMap() => {
        'id': id,
        'modo': 0,
        'titulo': title,
        'creado_en': createdAt.toIso8601String(),
        'actualizado_en': updatedAt.toIso8601String(),
      };

  factory Conversation.fromMap(Map<String, dynamic> map) => Conversation(
        id: map['id'] as String,
        title: map['titulo'] as String,
        createdAt: DateTime.parse(map['creado_en'] as String),
        updatedAt: DateTime.parse(map['actualizado_en'] as String),
      );
}
