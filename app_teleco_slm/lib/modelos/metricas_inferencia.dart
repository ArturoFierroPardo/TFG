/// Metrics for a single inference request.
class InferenceMetrics {
  final DateTime timestamp;
  final String userPrompt;
  final int tokensGenerated;
  final double timeToFirstTokenMs;
  final double totalTimeMs;
  final double tokensPerSecond;
  final double cpuUsagePercent;
  final String backend; // 'local', 'remote', 'fllama'

  InferenceMetrics({
    required this.timestamp,
    required this.userPrompt,
    required this.tokensGenerated,
    required this.timeToFirstTokenMs,
    required this.totalTimeMs,
    required this.tokensPerSecond,
    required this.cpuUsagePercent,
    required this.backend,
  });

  String get formattedSummary {
    return 'Tokens: $tokensGenerated | '
        'First token: ${timeToFirstTokenMs.toStringAsFixed(0)} ms | '
        'Total: ${(totalTimeMs / 1000).toStringAsFixed(1)} s | '
        'Speed: ${tokensPerSecond.toStringAsFixed(1)} tok/s | '
        'CPU: ${cpuUsagePercent.toStringAsFixed(1)}%';
  }
}
