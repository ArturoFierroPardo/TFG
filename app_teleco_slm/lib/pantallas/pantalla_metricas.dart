/// Performance metrics screen - shows inference stats per request.
library;

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../proveedores/proveedor_app.dart';
import '../utilidades/tema.dart';
import '../utilidades/traducciones.dart';

class PantallaMetricas extends StatelessWidget {
  const PantallaMetricas({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Tema.fondoBase,
      appBar: AppBar(
        backgroundColor: Tema.fondoBase,
        foregroundColor: Tema.textoBase,
        title: Text(Tr.get('inferenceMetrics'),
            style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w600)),
        elevation: 0,
      ),
      body: Consumer<AppProvider>(
        builder: (context, provider, _) {
          final metrics = provider.metricsHistory;

          if (metrics.isEmpty) {
            return Center(
              child: Text(
                Tr.get('noMetrics'),
                textAlign: TextAlign.center,
                style: const TextStyle(color: Tema.textoApagado, fontSize: 14),
              ),
            );
          }

          return ListView.builder(
            padding: const EdgeInsets.all(16),
            itemCount: metrics.length,
            itemBuilder: (context, index) {
              final m = metrics[index];
              return Container(
                margin: const EdgeInsets.only(bottom: 12),
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Tema.fondoTarjeta,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: Tema.borde, width: 0.5),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Icon(Icons.access_time_rounded, size: 14, color: Tema.textoApagado),
                        const SizedBox(width: 6),
                        Text(
                          '${m.timestamp.hour.toString().padLeft(2, '0')}:'
                          '${m.timestamp.minute.toString().padLeft(2, '0')}:'
                          '${m.timestamp.second.toString().padLeft(2, '0')}',
                          style: const TextStyle(fontSize: 12, color: Tema.textoApagado),
                        ),
                        const Spacer(),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                          decoration: BoxDecoration(
                            color: Tema.primario.withOpacity(0.15),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Text(
                            m.backend,
                            style: const TextStyle(fontSize: 10, color: Tema.primario, fontWeight: FontWeight.w600),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Text(
                      m.userPrompt,
                      style: const TextStyle(fontSize: 13, color: Tema.textoSecundario, fontStyle: FontStyle.italic),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 12),
                    _metricRow(Tr.get('tokensGenerated'), '${m.tokensGenerated}'),
                    _metricRow(Tr.get('timeToFirstToken'), '${m.timeToFirstTokenMs.toStringAsFixed(0)} ms'),
                    _metricRow(Tr.get('totalTime'), '${(m.totalTimeMs / 1000).toStringAsFixed(2)} s'),
                    _metricRow(Tr.get('speed'), '${m.tokensPerSecond.toStringAsFixed(1)} tok/s'),
                    _metricRow(Tr.get('memory'), '${m.cpuUsagePercent.toStringAsFixed(1)}%'),
                  ],
                ),
              );
            },
          );
        },
      ),
    );
  }

  Widget _metricRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontSize: 13, color: Tema.textoApagado)),
          Text(value, style: const TextStyle(fontSize: 13, color: Tema.textoBase, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}
