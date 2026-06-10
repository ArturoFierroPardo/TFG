/// SQLite database service.
library;

import 'dart:io';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';
import '../modelos/modelos_chat.dart';

class ServicioBD {
  static Database? _db;

  static Future<void> inicializar() async {
    if (Platform.isWindows || Platform.isLinux) {
      sqfliteFfiInit();
      databaseFactory = databaseFactoryFfi;
    }
  }

  static Future<Database> get database async {
    if (_db != null) return _db!;
    _db = await _createDB();
    return _db!;
  }

  static Future<Database> _createDB() async {
    final dir = await getApplicationDocumentsDirectory();
    final dbPath = p.join(dir.path, 'telecom_llm', 'historial.db');
    await Directory(p.dirname(dbPath)).create(recursive: true);

    return await openDatabase(
      dbPath,
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE conversaciones (
            id TEXT PRIMARY KEY,
            modo INTEGER NOT NULL,
            titulo TEXT NOT NULL,
            creado_en TEXT NOT NULL,
            actualizado_en TEXT NOT NULL
          )
        ''');
        await db.execute('''
          CREATE TABLE mensajes (
            id TEXT PRIMARY KEY,
            conversacion_id TEXT NOT NULL,
            rol TEXT NOT NULL,
            contenido TEXT NOT NULL,
            marca TEXT NOT NULL,
            FOREIGN KEY (conversacion_id) 
              REFERENCES conversaciones(id) ON DELETE CASCADE
          )
        ''');
        await db.execute(
          'CREATE INDEX idx_mensajes_conv ON mensajes(conversacion_id)',
        );
      },
    );
  }

  static const int maxChats = 20;

  Future<List<Conversation>> obtenerConversaciones() async {
    final db = await database;
    final rows = await db.query('conversaciones', orderBy: 'actualizado_en DESC');
    return rows.map(Conversation.fromMap).toList();
  }

  Future<Conversation> crearConversacion(Conversation conv) async {
    final db = await database;
    await db.insert('conversaciones', conv.toMap());
    // Auto-delete oldest chats beyond limit
    await _limpiarChatsAntiguos(db);
    return conv;
  }

  Future<void> _limpiarChatsAntiguos(Database db) async {
    final rows = await db.query('conversaciones',
        orderBy: 'actualizado_en DESC');
    if (rows.length > maxChats) {
      final toDelete = rows.sublist(maxChats);
      for (final row in toDelete) {
        final id = row['id'] as String;
        await db.delete('mensajes', where: 'conversacion_id = ?', whereArgs: [id]);
        await db.delete('conversaciones', where: 'id = ?', whereArgs: [id]);
      }
    }
  }

  Future<void> actualizarConversacion(Conversation conv) async {
    final db = await database;
    await db.update('conversaciones', conv.toMap(),
        where: 'id = ?', whereArgs: [conv.id]);
  }

  Future<void> eliminarConversacion(String id) async {
    final db = await database;
    await db.delete('mensajes', where: 'conversacion_id = ?', whereArgs: [id]);
    await db.delete('conversaciones', where: 'id = ?', whereArgs: [id]);
  }

  Future<List<ChatMessage>> obtenerMensajes(String conversationId) async {
    final db = await database;
    final rows = await db.query('mensajes',
        where: 'conversacion_id = ?',
        whereArgs: [conversationId],
        orderBy: 'marca ASC');
    return rows.map(ChatMessage.fromMap).toList();
  }

  Future<void> insertarMensaje(ChatMessage msg) async {
    final db = await database;
    await db.insert('mensajes', msg.toMap(),
        conflictAlgorithm: ConflictAlgorithm.replace);
  }

  Future<void> borrarTodoElHistorial() async {
    final db = await database;
    await db.delete('mensajes');
    await db.delete('conversaciones');
  }
}
