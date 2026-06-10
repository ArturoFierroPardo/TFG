/// Model status badge widget.
library;

import 'package:flutter/material.dart';
import '../proveedores/proveedor_app.dart';
import '../utilidades/tema.dart';
import '../utilidades/traducciones.dart';

class BadgeEstadoModelo extends StatelessWidget {
  final ModelStatus estado;
  final bool compacto;

  const BadgeEstadoModelo({
    super.key,
    required this.estado,
    this.compacto = false,
  });

  @override
  Widget build(BuildContext context) {
    final cfg = _getConfig();

    return AnimatedContainer(
      duration: Tema.duracionNormal,
      curve: Tema.curvaEntrada,
      padding: EdgeInsets.symmetric(
        horizontal: compacto ? 8 : 12,
        vertical: compacto ? 4 : 6,
      ),
      decoration: BoxDecoration(
        color: cfg.color.withOpacity(0.10),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: cfg.color.withOpacity(0.20), width: 0.5),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (estado == ModelStatus.loading)
            SizedBox(
              width: 12, height: 12,
              child: CircularProgressIndicator(strokeWidth: 1.5, color: cfg.color),
            )
          else
            Container(
              width: 7, height: 7,
              decoration: BoxDecoration(
                color: cfg.color,
                shape: BoxShape.circle,
                boxShadow: [BoxShadow(color: cfg.color.withOpacity(0.5), blurRadius: 6, spreadRadius: 1)],
              ),
            ),
          const SizedBox(width: 6),
          Text(
            cfg.label,
            style: TextStyle(
              color: cfg.color,
              fontSize: compacto ? 11 : 12,
              fontWeight: FontWeight.w500,
              letterSpacing: 0.2,
            ),
          ),
        ],
      ),
    );
  }

  _StatusConfig _getConfig() {
    switch (estado) {
      case ModelStatus.notFound:
        return _StatusConfig(Tr.isEs ? 'Sin modelo' : 'No model', Tema.textoApagado);
      case ModelStatus.found:
        return _StatusConfig(Tr.isEs ? 'Modelo listo' : 'Model ready', Tema.advertencia);
      case ModelStatus.loading:
        return _StatusConfig(Tr.isEs ? 'Cargando...' : 'Loading...', Tema.advertencia);
      case ModelStatus.loaded:
        return _StatusConfig(Tr.isEs ? 'Modelo activo' : 'Model active', Tema.exito);
      case ModelStatus.error:
        return _StatusConfig(Tr.isEs ? 'Modelo inactivo' : 'Model inactive', Tema.error);
    }
  }
}

class _StatusConfig {
  final String label;
  final Color color;
  const _StatusConfig(this.label, this.color);
}
