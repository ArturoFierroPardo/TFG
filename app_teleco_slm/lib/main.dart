/// Teleco SLM - Entry point.
library;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'proveedores/proveedor_app.dart';
import 'pantallas/pantalla_inicio.dart';
import 'servicios/servicio_bd.dart';
import 'utilidades/tema.dart';
import 'utilidades/traducciones.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ServicioBD.inicializar();

  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]);

  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.light,
    systemNavigationBarColor: Tema.fondoBase,
    systemNavigationBarIconBrightness: Brightness.light,
  ));

  runApp(const TelecoSlmApp());
}

class TelecoSlmApp extends StatelessWidget {
  const TelecoSlmApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => AppProvider()..initialize(),
      child: MaterialApp(
        title: 'Teleco SLM',
        debugShowCheckedModeBanner: false,
        theme: Tema.temaOscuro,
        home: const _SplashWrapper(),
      ),
    );
  }
}

class _SplashWrapper extends StatefulWidget {
  const _SplashWrapper();

  @override
  State<_SplashWrapper> createState() => _SplashWrapperState();
}

class _SplashWrapperState extends State<_SplashWrapper>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  bool _ready = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 1000),
      vsync: this,
    )..forward();

    Future.delayed(const Duration(milliseconds: 1600), () {
      if (mounted) setState(() => _ready = true);
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_ready) return const PantallaInicio();

    return Scaffold(
      backgroundColor: Tema.fondoBase,
      body: Center(
        child: FadeTransition(
          opacity: CurvedAnimation(
            parent: _controller,
            curve: const Interval(0.0, 0.6, curve: Tema.curvaEntrada),
          ),
          child: ScaleTransition(
            scale: Tween<double>(begin: 0.88, end: 1.0).animate(
              CurvedAnimation(
                parent: _controller,
                curve: const Interval(0.0, 0.6, curve: Tema.curvaEntrada),
              ),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      colors: [Tema.primario, Tema.acento],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    borderRadius: BorderRadius.circular(22),
                    boxShadow: [
                      BoxShadow(
                        color: Tema.primario.withOpacity(0.35),
                        blurRadius: 32,
                        spreadRadius: 4,
                      ),
                    ],
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(18),
                    child: Image.asset('assets/iconos/app_logo.png', width: 80, height: 80),
                  ),
                ),
                const SizedBox(height: 22),
                const Text(
                  'Teleco SLM',
                  style: TextStyle(fontSize: 28, fontWeight: FontWeight.w700, color: Tema.textoBase, letterSpacing: -0.5),
                ),
                const SizedBox(height: 8),
                Text(
                  Tr.get('splashSubtitle'),
                  style: const TextStyle(fontSize: 13, color: Tema.textoApagado, letterSpacing: 0.8),
                ),
                const SizedBox(height: 28),
                SizedBox(
                  width: 24,
                  height: 24,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: Tema.primarioClaro.withOpacity(0.5),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
