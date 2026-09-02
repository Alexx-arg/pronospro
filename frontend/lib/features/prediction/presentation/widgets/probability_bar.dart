import 'package:flutter/material.dart';

/// Barra individual — D14.1 Material 3 puro
class ProbabilityBar extends StatelessWidget {
  const ProbabilityBar({
    super.key,
    required this.label,
    required this.value,
    this.color,
  });

  final String label;
  final double value; // 0..1
  final Color? color;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: Theme.of(context).textTheme.bodyMedium),
            Text('${(value * 100).toStringAsFixed(1)}%',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.bold)),
          ],
        ),
        const SizedBox(height: 4),
        LinearProgressIndicator(
          value: value.clamp(0.0, 1.0),
          backgroundColor: Theme.of(context).colorScheme.surfaceContainerHighest,
          color: color ?? Theme.of(context).colorScheme.primary,
          minHeight: 8,
          borderRadius: BorderRadius.circular(4),
        ),
      ],
    );
  }
}

/// Card que agrupa las 3 probabilidades — usado en Success
class PredictionProbabilityCard extends StatelessWidget {
  const PredictionProbabilityCard({
    super.key,
    required this.probHome,
    required this.probDraw,
    required this.probAway,
    required this.modelVersion,
    this.fixtureId,
  });

  final double probHome;
  final double probDraw;
  final double probAway;
  final String modelVersion;
  final int? fixtureId;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      elevation: 1,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (fixtureId != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Text('Fixture #$fixtureId',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
              ),
            ProbabilityBar(label: 'Local', value: probHome, color: scheme.primary),
            const SizedBox(height: 12),
            ProbabilityBar(label: 'Empate', value: probDraw, color: scheme.secondary),
            const SizedBox(height: 12),
            ProbabilityBar(label: 'Visitante', value: probAway, color: scheme.tertiary),
            const SizedBox(height: 16),
            Text('Modelo: $modelVersion',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey),
                textAlign: TextAlign.end),
          ],
        ),
      ),
    );
  }
}
