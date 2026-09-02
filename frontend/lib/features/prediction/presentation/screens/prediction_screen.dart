import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:intl/intl.dart';

import 'package:football_prediction_app/core/network/api_client.dart';
import 'package:football_prediction_app/features/prediction/prediction.dart';
import 'package:football_prediction_app/features/fixtures/fixtures.dart';

/// Prediction Detail Screen — shows match details, metrics, probabilities, and AI analysis.
class PredictionScreen extends ConsumerStatefulWidget {
  const PredictionScreen({super.key, this.fixtureId});

  final int? fixtureId;

  @override
  ConsumerState<PredictionScreen> createState() => _PredictionScreenState();
}

class _PredictionScreenState extends ConsumerState<PredictionScreen> {
  FixtureResponse? _fixture;
  bool _loadingFixture = false;

  @override
  void initState() {
    super.initState();
    if (widget.fixtureId != null) {
      _loadFixtureDetail(widget.fixtureId!);
    }
  }

  Future<void> _loadFixtureDetail(int fixtureId) async {
    setState(() => _loadingFixture = true);
    final apiClient = ref.read(apiClientProvider);
    try {
      final response = await apiClient.dio.get(
        '/api/v1/fixtures/$fixtureId',
        queryParameters: {
          'include_metrics': true,
          'include_prediction': true,
        },
      );
      if (response.statusCode == 200) {
        setState(() => _fixture = FixtureResponse.fromJson(response.data as Map<String, dynamic>));
      }
    } catch (e) {
      debugPrint('Error loading fixture: $e');
    } finally {
      setState(() => _loadingFixture = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(predictionNotifierProvider);
    final scheme = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: Text(widget.fixtureId != null ? 'Predicción' : 'Predicción 1X2'),
      ),
      body: _loadingFixture
          ? const Center(child: CircularProgressIndicator())
          : widget.fixtureId != null && _fixture != null
              ? _buildFixtureDetail(context, _fixture!, scheme)
              : _buildManualInput(context, state, scheme),
    );
  }

  Widget _buildManualInput(BuildContext context, PredictionState state, ColorScheme scheme) {
    final _controller = TextEditingController();
    final _formKey = GlobalKey<FormState>();

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Form(
        key: _formKey,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextFormField(
              controller: _controller,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(
                labelText: 'Fixture ID',
                hintText: 'Ej: 1001',
                border: OutlineInputBorder(),
              ),
              validator: (v) {
                if (v == null || v.isEmpty) return 'Ingrese un ID';
                if (int.tryParse(v) == null) return 'Debe ser numérico';
                return null;
              },
            ),
            const SizedBox(height: 12),
            ElevatedButton(
              onPressed: state is PredictionLoading
                  ? null
                  : () {
                      FocusScope.of(context).unfocus();
                      if (_formKey.currentState!.validate()) {
                        final id = int.parse(_controller.text);
                        ref.read(predictionNotifierProvider.notifier).fetchPrediction(id);
                      }
                    },
              child: const Text('Predecir'),
            ),
            const SizedBox(height: 24),
            _buildPredictionResult(state, scheme),
          ],
        ),
      ),
    );
  }

  Widget _buildFixtureDetail(BuildContext context, FixtureResponse fixture, ColorScheme scheme) {
    final probHome = fixture.prediction?.homeProbability;
    final probDraw = fixture.prediction?.drawProbability;
    final probAway = fixture.prediction?.awayProbability;

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // Match header with logos
        Row(
          children: [
            _TeamLogo(team: fixture.homeTeam, size: 60),
            const Spacer(),
            Column(
              children: [
                Text(
                  'vs',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
                ),
                Text(
                  DateFormat('EEE dd MMM • HH:mm', 'es').format(fixture.kickoffTime.toLocal()),
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
                ),
              ],
            ),
            const Spacer(),
            _TeamLogo(team: fixture.awayTeam, size: 60),
          ],
        ),
        if (fixture.venue != null) ...[
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.location_on, size: 16, color: scheme.onSurfaceVariant),
              const SizedBox(width: 4),
              Text(fixture.venue!, style: Theme.of(context).textTheme.bodyMedium),
            ],
          ),
        ],
        const SizedBox(height: 24),
        // Competition badge
        Center(
          child: Chip(
            avatar: fixture.competition.logo != null
                ? Image.network(fixture.competition.logo!, width: 20, height: 20)
                : null,
            label: Text(fixture.competition.name),
            backgroundColor: scheme.primaryContainer,
          ),
        ),
        const SizedBox(height: 24),
        // Prediction
        if (probHome != null && probDraw != null && probAway != null) ...[
          Text('Predicción 1X2', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
          const SizedBox(height: 12),
          PredictionProbabilityCard(
            fixtureId: fixture.id,
            probHome: probHome,
            probDraw: probDraw,
            probAway: probAway,
            modelVersion: fixture.prediction?.modelVersion ?? 'unknown',
          ),
          const SizedBox(height: 12),
          NvidiaExplainButton(
            fixtureId: fixture.id,
            probHome: probHome,
            probDraw: probDraw,
            probAway: probAway,
            homeTeam: fixture.homeTeam.name,
            awayTeam: fixture.awayTeam.name,
            metrics: fixture.metrics,
          ),
        ] else ...[
          const Center(child: Text('Sin predicción disponible para este partido')),
          const SizedBox(height: 12),
          FilledButton.icon(
            icon: const Icon(Icons.auto_awesome),
            label: const Text('Generar Predicción'),
            onPressed: () {
              ref.read(predictionNotifierProvider.notifier).fetchPrediction(fixture.id);
            },
          ),
        ],
        const SizedBox(height: 24),
        // Metrics
        if (fixture.metrics != null) ...[
          Text('Métricas del Partido', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
          const SizedBox(height: 12),
          _MetricsCard(metrics: fixture.metrics!, homeTeam: fixture.homeTeam.name, awayTeam: fixture.awayTeam.name, scheme: scheme),
        ],
      ],
    );
  }

  Widget _buildPredictionResult(PredictionState state, ColorScheme scheme) {
    return switch (state) {
      PredictionInitial() => const Text('Ingrese un fixture y presione Predecir', textAlign: TextAlign.center),
      PredictionLoading() => const Center(child: CircularProgressIndicator()),
      PredictionSuccess(:final response) => Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            PredictionProbabilityCard(
              fixtureId: response.fixtureId,
              probHome: response.probHome,
              probDraw: response.probDraw,
              probAway: response.probAway,
              modelVersion: response.modelVersion,
            ),
            const SizedBox(height: 12),
            // Note: AI explain needs fixture context, only available when loaded via fixtures list
            const Text(
              'Abra el partido desde la lista "Próximos Partidos" para ver métricas y análisis IA.',
              style: TextStyle(fontSize: 12, color: Colors.grey),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      PredictionError(:final message) => Text(
          message,
          style: const TextStyle(color: Colors.red),
          textAlign: TextAlign.center,
        ),
    };
  }
}

class _TeamLogo extends StatelessWidget {
  const _TeamLogo({required this.team, required this.size});
  final TeamInfo team;
  final double size;

  @override
  Widget build(BuildContext context) {
    final color = Theme.of(context).colorScheme.surfaceContainerHighest;
    if (team.logo != null) {
      return CachedNetworkImage(
        imageUrl: team.logo!,
        width: size,
        height: size,
        placeholder: (_, __) => _placeholder(size, color),
        errorWidget: (_, __, ___) => _placeholder(size, color),
      );
    }
    return _placeholder(size, color);
  }

  Widget _placeholder(double size, Color color) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: color,
        shape: BoxShape.circle,
      ),
      child: const Icon(Icons.shield, size: 24),
    );
  }
}

class _MetricsCard extends StatelessWidget {
  const _MetricsCard({
    required this.metrics,
    required this.homeTeam,
    required this.awayTeam,
    required this.scheme,
  });

  final MatchMetrics metrics;
  final String homeTeam;
  final String awayTeam;
  final ColorScheme scheme;

  @override
  Widget build(BuildContext context) {
    final items = <_MetricRow>[
      if (metrics.homeForm != null || metrics.awayForm != null)
        _MetricRow('Forma', metrics.homeForm ?? '-', metrics.awayForm ?? '-'),
      if (metrics.homeXg != null || metrics.awayXg != null)
        _MetricRow('xG', metrics.homeXg!.toStringAsFixed(2), metrics.awayXg!.toStringAsFixed(2)),
      if (metrics.homeXga != null || metrics.awayXga != null)
        _MetricRow('xGA', metrics.homeXga!.toStringAsFixed(2), metrics.awayXga!.toStringAsFixed(2)),
      if (metrics.homeCornersAvg != null || metrics.awayCornersAvg != null)
        _MetricRow('Córners/partido', metrics.homeCornersAvg!.toStringAsFixed(1), metrics.awayCornersAvg!.toStringAsFixed(1)),
      if (metrics.homeYellowCardsAvg != null || metrics.awayYellowCardsAvg != null)
        _MetricRow('Amarillas/partido', metrics.homeYellowCardsAvg!.toStringAsFixed(1), metrics.awayYellowCardsAvg!.toStringAsFixed(1)),
      if (metrics.homePossessionAvg != null || metrics.awayPossessionAvg != null)
        _MetricRow('Posesión %', '${metrics.homePossessionAvg!.toStringAsFixed(1)}%', '${metrics.awayPossessionAvg!.toStringAsFixed(1)}%'),
    ];

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
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
            const SizedBox(height: 12),
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
        ),
      ),
    );
  }
}

class _MetricRow {
  const _MetricRow(this.label, this.homeValue, this.awayValue);
  final String label;
  final String homeValue;
  final String awayValue;
}