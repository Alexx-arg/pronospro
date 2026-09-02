import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:dio/dio.dart';

import 'package:football_prediction_app/core/network/api_client.dart';
import 'package:football_prediction_app/features/fixtures/fixtures.dart';

class NvidiaExplainButton extends ConsumerStatefulWidget {
  const NvidiaExplainButton({
    super.key,
    required this.fixtureId,
    required this.probHome,
    required this.probDraw,
    required this.probAway,
    required this.homeTeam,
    required this.awayTeam,
    this.metrics,
  });

  final int fixtureId;
  final double probHome;
  final double probDraw;
  final double probAway;
  final String homeTeam;
  final String awayTeam;
  final MatchMetrics? metrics;

  @override
  ConsumerState<NvidiaExplainButton> createState() => _NvidiaExplainButtonState();
}

class _NvidiaExplainButtonState extends ConsumerState<NvidiaExplainButton> {
  String? _explanation;
  bool _loading = false;

  Future<void> _explain() async {
    setState(() {
      _loading = true;
      _explanation = null;
    });

    final apiClient = ref.read(apiClientProvider);
    try {
      final response = await apiClient.dio.post(
        '/api/v1/explain',
        data: {
          'fixture_id': widget.fixtureId,
          'prob_home': widget.probHome,
          'prob_draw': widget.probDraw,
          'prob_away': widget.probAway,
          'home_team': widget.homeTeam,
          'away_team': widget.awayTeam,
          'metrics': widget.metrics?.toJson(),
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

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        FilledButton.icon(
          onPressed: _loading ? null : _explain,
          icon: _loading
              ? const SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                )
              : const Icon(Icons.auto_awesome),
          label: Text(_loading ? 'Analizando con IA...' : '✨ Analizar con IA (NVIDIA)'),
          style: FilledButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 14),
            backgroundColor: scheme.tertiaryContainer,
            foregroundColor: scheme.onTertiaryContainer,
          ),
        ),
        if (_explanation != null) ...[
          const SizedBox(height: 12),
          Card(
            color: scheme.surfaceContainerHighest,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.auto_awesome, size: 18, color: scheme.tertiary),
                      const SizedBox(width: 8),
                      Text(
                        'Análisis IA (NVIDIA NIM)',
                        style: Theme.of(context).textTheme.labelLarge?.copyWith(
                          color: scheme.tertiary,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(_explanation!, style: Theme.of(context).textTheme.bodyMedium),
                ],
              ),
            ),
          ),
        ],
      ],
    );
  }
}