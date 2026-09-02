import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import 'package:football_prediction_app/features/fixtures/fixtures.dart';
import 'package:football_prediction_app/core/network/api_client.dart';

class FixtureCard extends StatelessWidget {
  const FixtureCard({
    super.key,
    required this.fixture,
  });

  final FixtureResponse fixture;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;

    final probHome = fixture.prediction?.homeProbability;
    final probDraw = fixture.prediction?.drawProbability;
    final probAway = fixture.prediction?.awayProbability;

    return Card(
      elevation: 1,
      margin: const EdgeInsets.only(bottom: 12),
      child: InkWell(
        onTap: () => _showDetail(context),
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // League and time
              Row(
                children: [
                  if (fixture.competition.logo != null)
                    Image.network(
                      fixture.competition.logo!,
                      width: 20,
                      height: 20,
                      errorBuilder: (_, __, ___) =>
                          const Icon(Icons.sports_soccer, size: 20),
                    )
                  else
                    const Icon(Icons.sports_soccer, size: 20),
                  const SizedBox(width: 8),
                  Text(
                    fixture.competition.name,
                    style: textTheme.labelLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: scheme.primary,
                    ),
                  ),
                  const Spacer(),
                  Text(
                    DateFormat('EEE dd MMM • HH:mm', 'es').format(fixture.kickoffTime.toLocal()),
                    style: textTheme.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              // Teams
              Row(
                children: [
                  _TeamWidget(team: fixture.homeTeam, isHome: true),
                  const Spacer(),
                  const Text('vs', style: TextStyle(color: Colors.grey, fontWeight: FontWeight.bold)),
                  const Spacer(),
                  _TeamWidget(team: fixture.awayTeam, isHome: false),
                ],
              ),
              if (fixture.venue != null) ...[
                const SizedBox(height: 8),
                Row(
                  children: [
                    Icon(Icons.location_on, size: 14, color: scheme.onSurfaceVariant),
                    const SizedBox(width: 4),
                    Expanded(
                      child: Text(
                        fixture.venue!,
                        style: textTheme.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ],
                ),
              ],
              // Probabilities if available
              if (probHome != null && probDraw != null && probAway != null) ...[
                const SizedBox(height: 12),
                _ProbabilityRow(
                  homeLabel: fixture.homeTeam.shortName ?? fixture.homeTeam.name,
                  homeProb: probHome,
                  drawProb: probDraw,
                  awayProb: probAway,
                  awayLabel: fixture.awayTeam.shortName ?? fixture.awayTeam.name,
                  modelVersion: fixture.prediction?.modelVersion,
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  void _showDetail(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => _FixtureDetailBottomSheet(fixture: fixture),
    );
  }
}

class _TeamWidget extends StatelessWidget {
  const _TeamWidget({
    required this.team,
    required this.isHome,
  });

  final TeamInfo team;
  final bool isHome;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 80,
      child: Column(
        children: [
          if (team.logo != null)
            Image.network(
              team.logo!,
              width: 40,
              height: 40,
              errorBuilder: (_, __, ___) => Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surfaceContainerHighest,
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.shield, size: 24),
              ),
            )
          else
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surfaceContainerHighest,
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.shield, size: 24),
            ),
          const SizedBox(height: 4),
          Text(
            team.shortName ?? team.name,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              fontWeight: FontWeight.bold,
            ),
            textAlign: TextAlign.center,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }
}

class _ProbabilityRow extends StatelessWidget {
  const _ProbabilityRow({
    required this.homeLabel,
    required this.homeProb,
    required this.drawProb,
    required this.awayProb,
    required this.awayLabel,
    this.modelVersion,
  });

  final String homeLabel;
  final double homeProb;
  final double drawProb;
  final double awayProb;
  final String awayLabel;
  final String? modelVersion;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final maxProb = [homeProb, drawProb, awayProb].reduce((a, b) => a > b ? a : b);

    return Column(
      children: [
        Row(
          children: [
            Expanded(
              child: _ProbBar(
                label: homeLabel,
                prob: homeProb,
                isFavorite: homeProb == maxProb,
                color: scheme.primary,
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _ProbBar(
                label: 'X',
                prob: drawProb,
                isFavorite: drawProb == maxProb,
                color: scheme.secondary,
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _ProbBar(
                label: awayLabel,
                prob: awayProb,
                isFavorite: awayProb == maxProb,
                color: scheme.tertiary,
              ),
            ),
          ],
        ),
        if (modelVersion != null) ...[
          const SizedBox(height: 4),
          Text(
            'Modelo: $modelVersion',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: Colors.grey,
            ),
            textAlign: TextAlign.end,
          ),
        ],
      ],
    );
  }
}

class _ProbBar extends StatelessWidget {
  const _ProbBar({
    required this.label,
    required this.prob,
    required this.isFavorite,
    required this.color,
  });

  final String label;
  final double prob;
  final bool isFavorite;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: isFavorite ? color.withOpacity(0.1) : Colors.transparent,
        borderRadius: BorderRadius.circular(8),
        border: isFavorite ? Border.all(color: color, width: 1.5) : null,
      ),
      child: Column(
        children: [
          Text(
            label,
            style: Theme.of(context).textTheme.labelSmall?.copyWith(
              fontWeight: FontWeight.bold,
              color: isFavorite ? color : null,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 4),
          LinearProgressIndicator(
            value: prob.clamp(0.0, 1.0),
            backgroundColor: Theme.of(context).colorScheme.surfaceContainerHighest,
            color: color,
            minHeight: 6,
            borderRadius: BorderRadius.circular(3),
          ),
          const SizedBox(height: 4),
          Text(
            '${(prob * 100).toStringAsFixed(1)}%',
            style: Theme.of(context).textTheme.labelSmall?.copyWith(
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
}

class _FixtureDetailBottomSheet extends StatelessWidget {
  const _FixtureDetailBottomSheet({
    required this.fixture,
  });

  final FixtureResponse fixture;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;

    return Container(
      height: MediaQuery.of(context).size.height * 0.85,
      decoration: BoxDecoration(
        color: scheme.surface,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
      ),
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: scheme.outline,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            const SizedBox(height: 20),
            // Match header
            Row(
              children: [
                _TeamWidget(team: fixture.homeTeam, isHome: true),
                const Spacer(),
                Column(
                  children: [
                    Text(
                      'vs',
                      style: textTheme.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
                    ),
                    Text(
                      DateFormat('EEE dd MMM • HH:mm', 'es').format(fixture.kickoffTime.toLocal()),
                      style: textTheme.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
                    ),
                  ],
                ),
                const Spacer(),
                _TeamWidget(team: fixture.awayTeam, isHome: false),
              ],
            ),
            if (fixture.venue != null) ...[
              const SizedBox(height: 8),
              Row(
                children: [
                  Icon(Icons.location_on, size: 16, color: scheme.onSurfaceVariant),
                  const SizedBox(width: 4),
                  Text(fixture.venue!, style: textTheme.bodyMedium),
                ],
              ),
            ],
            const SizedBox(height: 24),
            // Prediction
            if (fixture.prediction != null) ...[
              Text('Predicción', style: textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
              const SizedBox(height: 12),
              _PredictionBars(prediction: fixture.prediction!),
              const SizedBox(height: 24),
            ],
            // Metrics
            if (fixture.metrics != null) ...[
              Text('Métricas del Partido', style: textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
              const SizedBox(height: 12),
              _MetricsGrid(metrics: fixture.metrics!, homeTeam: fixture.homeTeam.name, awayTeam: fixture.awayTeam.name),
              const SizedBox(height: 24),
            ],
            // AI Explain button
            if (fixture.prediction != null) ...[
              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  icon: const Icon(Icons.auto_awesome),
                  label: const Text('Analizar con IA (NVIDIA)'),
                  onPressed: () => _showExplanation(context),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  void _showExplanation(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => _ExplanationBottomSheet(fixture: fixture),
    );
  }
}

class _PredictionBars extends StatelessWidget {
  const _PredictionBars({required this.prediction});

  final PredictionInfo prediction;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final probs = [
      ('Local', prediction.homeProbability, scheme.primary),
      ('Empate', prediction.drawProbability, scheme.secondary),
      ('Visitante', prediction.awayProbability, scheme.tertiary),
    ];
    final maxProb = probs.reduce((a, b) => a.$2 > b.$2 ? a : b).$2;

    return Column(
      children: probs.map((p) {
        final isFav = p.$2 == maxProb;
        return Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: isFav ? p.$3.withOpacity(0.1) : Colors.transparent,
              borderRadius: BorderRadius.circular(8),
              border: isFav ? Border.all(color: p.$3, width: 1.5) : null,
            ),
            child: Row(
              children: [
                Expanded(
                  child: Text(p.$1, style: TextStyle(fontWeight: FontWeight.bold, color: isFav ? p.$3 : null)),
                ),
                LinearProgressIndicator(
                  value: p.$2.clamp(0.0, 1.0),
                  backgroundColor: scheme.surfaceContainerHighest,
                  color: p.$3,
                  minHeight: 8,
                  borderRadius: BorderRadius.circular(4),
                ),
                const SizedBox(width: 12),
                Text('${(p.$2 * 100).toStringAsFixed(1)}%',
                    style: TextStyle(fontWeight: FontWeight.bold, color: p.$3)),
              ],
            ),
          ),
        );
      }).toList(),
    );
  }
}

class _MetricsGrid extends StatelessWidget {
  const _MetricsGrid({
    required this.metrics,
    required this.homeTeam,
    required this.awayTeam,
  });

  final MatchMetrics metrics;
  final String homeTeam;
  final String awayTeam;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;

    final items = <_MetricItem>[
      if (metrics.homeForm != null || metrics.awayForm != null)
        _MetricItem('Forma', metrics.homeForm ?? '-', metrics.awayForm ?? '-'),
      if (metrics.homeXg != null || metrics.awayXg != null)
        _MetricItem('xG', metrics.homeXg!.toStringAsFixed(2), metrics.awayXg!.toStringAsFixed(2)),
      if (metrics.homeXga != null || metrics.awayXga != null)
        _MetricItem('xGA', metrics.homeXga!.toStringAsFixed(2), metrics.awayXga!.toStringAsFixed(2)),
      if (metrics.homeCornersAvg != null || metrics.awayCornersAvg != null)
        _MetricItem('Córners/partido', metrics.homeCornersAvg!.toStringAsFixed(1), metrics.awayCornersAvg!.toStringAsFixed(1)),
      if (metrics.homeYellowCardsAvg != null || metrics.awayYellowCardsAvg != null)
        _MetricItem('Amarillas/partido', metrics.homeYellowCardsAvg!.toStringAsFixed(1), metrics.awayYellowCardsAvg!.toStringAsFixed(1)),
      if (metrics.homePossessionAvg != null || metrics.awayPossessionAvg != null)
        _MetricItem('Posesión %', '${metrics.homePossessionAvg!.toStringAsFixed(1)}%', '${metrics.awayPossessionAvg!.toStringAsFixed(1)}%'),
    ];

    return Column(
      children: [
        Row(
          children: [
            const Expanded(child: SizedBox()),
            Text(homeTeam, style: TextStyle(fontWeight: FontWeight.bold, color: scheme.primary)),
            const SizedBox(width: 16),
            Text(awayTeam, style: TextStyle(fontWeight: FontWeight.bold, color: scheme.tertiary)),
            const Expanded(child: SizedBox()),
          ],
        ),
        const SizedBox(height: 8),
        ...items.map((item) => Padding(
          padding: const EdgeInsets.symmetric(vertical: 6),
          child: Row(
            children: [
              const Expanded(child: SizedBox()),
              Text(item.label, style: TextStyle(color: scheme.onSurfaceVariant)),
              const SizedBox(width: 16),
              Text(item.homeValue, style: TextStyle(fontWeight: FontWeight.bold, color: scheme.primary)),
              const SizedBox(width: 16),
              Text(item.awayValue, style: TextStyle(fontWeight: FontWeight.bold, color: scheme.tertiary)),
              const Expanded(child: SizedBox()),
            ],
          ),
        )),
      ],
    );
  }
}

class _MetricItem {
  const _MetricItem(this.label, this.homeValue, this.awayValue);
  final String label;
  final String homeValue;
  final String awayValue;
}

class _ExplanationBottomSheet extends ConsumerStatefulWidget {
  const _ExplanationBottomSheet({required this.fixture});
  final FixtureResponse fixture;

  @override
  ConsumerState<_ExplanationBottomSheet> createState() => _ExplanationBottomSheetState();
}

class _ExplanationBottomSheetState extends ConsumerState<_ExplanationBottomSheet> {
  String? _explanation;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _fetchExplanation();
  }

  Future<void> _fetchExplanation() async {
    setState(() => _loading = true);
    final apiClient = ref.read(apiClientProvider);
    try {
      final prediction = widget.fixture.prediction!;
      final response = await apiClient.dio.post(
        '/api/v1/explain',
        data: {
          'fixture_id': widget.fixture.id,
          'prob_home': prediction.homeProbability,
          'prob_draw': prediction.drawProbability,
          'prob_away': prediction.awayProbability,
          'home_team': widget.fixture.homeTeam.name,
          'away_team': widget.fixture.awayTeam.name,
          'metrics': widget.fixture.metrics?.toJson(),
        },
      );
      if (response.statusCode == 200) {
        setState(() => _explanation = response.data['explanation'] as String?);
      } else {
        setState(() => _explanation = 'Error: ${response.data['detail'] ?? 'Desconocido'}');
      }
    } catch (e) {
      setState(() => _explanation = 'Error de conexión: $e');
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;

    return Container(
      height: MediaQuery.of(context).size.height * 0.7,
      decoration: BoxDecoration(
        color: scheme.surface,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Text('Análisis IA', style: textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold)),
                const Spacer(),
                IconButton(
                  icon: const Icon(Icons.close),
                  onPressed: () => Navigator.pop(context),
                ),
              ],
            ),
            const SizedBox(height: 16),
            if (_loading)
              const Center(child: CircularProgressIndicator())
            else if (_explanation != null)
              Expanded(
                child: SingleChildScrollView(
                  child: Text(_explanation!, style: textTheme.bodyLarge),
                ),
              )
            else
              Text('No se pudo generar la explicación', style: textTheme.bodyMedium?.copyWith(color: Colors.red)),
          ],
        ),
      ),
    );
  }
}