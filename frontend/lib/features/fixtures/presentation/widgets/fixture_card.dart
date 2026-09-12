import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import 'package:football_prediction_app/core/network/api_client.dart';
import 'package:football_prediction_app/features/fixtures/fixtures.dart';

/// Enumeración de mercados a visualizar en la tarjeta.
enum FixtureMarket { w1x2, goals, metrics }

class FixtureCard extends StatefulWidget {
  const FixtureCard({super.key, required this.fixture});

  final FixtureResponse fixture;

  @override
  State<FixtureCard> createState() => _FixtureCardState();
}

class _FixtureCardState extends State<FixtureCard> {
  FixtureMarket _market = FixtureMarket.w1x2;

  @override
  Widget build(BuildContext context) {
    final fixture = widget.fixture;
    final scheme = Theme.of(context).colorScheme;

    return Card(
      elevation: 1,
      margin: const EdgeInsets.only(bottom: 10),
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: () => _showDetail(context),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(12, 10, 12, 12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Time + venue + status
              _TopRow(fixture: fixture),
              const SizedBox(height: 10),
              // Teams with logos
              Row(
                children: [
                  _TeamColumn(team: fixture.homeTeam),
                  Expanded(
                    child: _CenterScore(fixture: fixture),
                  ),
                  _TeamColumn(team: fixture.awayTeam),
                ],
              ),
              const SizedBox(height: 10),
              // Market selector chips
              _MarketSelector(market: _market, onChanged: _setMarket, scheme: scheme),
              const SizedBox(height: 8),
              // Market body
              _buildMarketBody(fixture, scheme),
            ],
          ),
        ),
      ),
    );
  }

  void _setMarket(FixtureMarket m) => setState(() => _market = m);

  Widget _buildMarketBody(FixtureResponse fixture, ColorScheme scheme) {
    final markets = fixture.markets;
    switch (_market) {
      case FixtureMarket.w1x2:
        final w = markets?.w1x2;
        if (w == null) return const _NoMarket('Sin predicción 1X2 todavía');
        return _W1X2Body(w1x2: w, scheme: scheme);
      case FixtureMarket.goals:
        final g = markets?.goals;
        if (g == null || (g.overUnder25 == null && g.btts == null)) {
          return const _NoMarket('Sin mercados de goles todavía');
        }
        return _GoalsBody(goals: g, scheme: scheme);
      case FixtureMarket.metrics:
        final m = markets?.metrics;
        if (m == null) return const _NoMarket('Sin métricas todavía');
        return _MetricsBody(metrics: m, scheme: scheme);
    }
  }

  void _showDetail(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => FixtureDetailSheet(fixture: widget.fixture),
    );
  }
}

class _TopRow extends StatelessWidget {
  const _TopRow({required this.fixture});
  final FixtureResponse fixture;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final time = DateFormat('HH:mm', 'es').format(fixture.kickoffTime.toLocal());
    return Row(
      children: [
        Text(
          time,
          style: Theme.of(context).textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.w700,
                color: scheme.primary,
              ),
        ),
        const SizedBox(width: 8),
        if (fixture.venue != null)
          Expanded(
            child: Text(
              fixture.venue!,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
              overflow: TextOverflow.ellipsis,
            ),
          ),
      ],
    );
  }
}

/// Logo con caché + fallback elegante.
class TeamLogo extends StatelessWidget {
  const TeamLogo({super.key, required this.logo, this.size = 40});

  final String? logo;
  final double size;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    if (logo != null && logo!.isNotEmpty) {
      return CachedNetworkImage(
        imageUrl: logo!,
        width: size,
        height: size,
        fit: BoxFit.contain,
        placeholder: (_, __) => _fallback(scheme),
        errorWidget: (_, __, ___) => _fallback(scheme),
      );
    }
    return _fallback(scheme);
  }

  Widget _fallback(ColorScheme scheme) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: scheme.surfaceContainerHighest,
        shape: BoxShape.circle,
      ),
      child: Icon(Icons.shield, size: size * 0.55, color: scheme.outline),
    );
  }
}

class _TeamColumn extends StatelessWidget {
  const _TeamColumn({required this.team});
  final TeamInfo team;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 76,
      child: Column(
        children: [
          TeamLogo(logo: team.logo, size: 40),
          const SizedBox(height: 6),
          Text(
            team.shortName ?? team.name,
            style: Theme.of(context).textTheme.labelMedium?.copyWith(fontWeight: FontWeight.w600),
            textAlign: TextAlign.center,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }
}

class _CenterScore extends StatelessWidget {
  const _CenterScore({required this.fixture});
  final FixtureResponse fixture;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    if (fixture.status == 'finished' &&
        fixture.homeGoals != null &&
        fixture.awayGoals != null) {
      return Text(
        '${fixture.homeGoals} - ${fixture.awayGoals}',
        style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800),
      );
    }
    return Text(
      'vs',
      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
            color: scheme.onSurfaceVariant,
            fontWeight: FontWeight.w600,
          ),
    );
  }
}

class _MarketSelector extends StatelessWidget {
  const _MarketSelector({
    required this.market,
    required this.onChanged,
    required this.scheme,
  });

  final FixtureMarket market;
  final void Function(FixtureMarket) onChanged;
  final ColorScheme scheme;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        _chip(context, FixtureMarket.w1x2, '1X2', Icons.sports),
        const SizedBox(width: 6),
        _chip(context, FixtureMarket.goals, 'Goles', Icons.sports_soccer),
        const SizedBox(width: 6),
        _chip(context, FixtureMarket.metrics, 'Métricas', Icons.query_stats),
      ],
    );
  }

  Widget _chip(BuildContext context, FixtureMarket m, String label, IconData icon) {
    final selected = m == market;
    return GestureDetector(
      onTap: () => onChanged(m),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: selected ? scheme.secondaryContainer : scheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 16, color: selected ? scheme.onSecondaryContainer : scheme.onSurfaceVariant),
            const SizedBox(width: 4),
            Text(
              label,
              style: Theme.of(context).textTheme.labelMedium?.copyWith(
                    color: selected ? scheme.onSecondaryContainer : scheme.onSurfaceVariant,
                    fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
                  ),
            ),
          ],
        ),
      ),
    );
  }
}

class _W1X2Body extends StatelessWidget {
  const _W1X2Body({required this.w1x2, required this.scheme});
  final W1X2Market w1x2;
  final ColorScheme scheme;

  @override
  Widget build(BuildContext context) {
    final probs = [
      ('1', w1x2.home, scheme.primary),
      ('X', w1x2.draw, scheme.secondary),
      ('2', w1x2.away, scheme.tertiary),
    ];
    final maxProb = [w1x2.home, w1x2.draw, w1x2.away].reduce((a, b) => a > b ? a : b);
    return Row(
      children: probs.map((p) {
        final fav = p.$2 == maxProb;
        return Expanded(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 3),
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 8),
              decoration: BoxDecoration(
                color: fav ? p.$3.withOpacity(0.12) : Colors.transparent,
                borderRadius: BorderRadius.circular(10),
                border: fav ? Border.all(color: p.$3, width: 1.5) : null,
              ),
              child: Column(
                children: [
                  Text('${(p.$2 * 100).toStringAsFixed(0)}%',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.w800,
                            color: fav ? p.$3 : scheme.onSurface,
                          )),
                  Text(p.$1,
                      style: Theme.of(context).textTheme.labelSmall?.copyWith(color: scheme.onSurfaceVariant)),
                ],
              ),
            ),
          ),
        );
      }).toList(),
    );
  }
}

class _GoalsBody extends StatelessWidget {
  const _GoalsBody({required this.goals, required this.scheme});
  final GoalsMarket goals;
  final ColorScheme scheme;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        if (goals.overUnder25 != null) _MarketBarPair(
          label: 'Over 2.5',
          a: goals.overUnder25!.over,
          b: goals.overUnder25!.under,
          bLabel: 'Under 2.5',
          leftColor: scheme.primary,
          rightColor: scheme.outline,
        ),
        const SizedBox(height: 8),
        if (goals.btts != null) _MarketBarPair(
          label: 'BTTS Sí',
          a: goals.btts!.yes,
          b: goals.btts!.no,
          bLabel: 'No',
          leftColor: scheme.tertiary,
          rightColor: scheme.outline,
        ),
      ],
    );
  }
}

class _MarketBarPair extends StatelessWidget {
  const _MarketBarPair({
    required this.label,
    required this.a,
    required this.b,
    required this.bLabel,
    required this.leftColor,
    required this.rightColor,
  });

  final String label;
  final double a;
  final double b;
  final String bLabel;
  final Color leftColor;
  final Color rightColor;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final aPct = (a * 100).toStringAsFixed(0);
    final bPct = (b * 100).toStringAsFixed(0);
    return Row(
      children: [
        Expanded(
          child: Row(
            children: [
              Icon(Icons.trending_up, size: 16, color: leftColor),
              const SizedBox(width: 4),
              Text('$label $aPct%',
                  style: TextStyle(fontWeight: FontWeight.w700, color: leftColor, fontSize: 13)),
            ],
          ),
        ),
        Expanded(
          flex: 2,
          child: Stack(
            children: [
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: a.clamp(0.0, 1.0),
                  minHeight: 8,
                  backgroundColor: scheme.surfaceContainerHighest,
                  valueColor: AlwaysStoppedAnimation(leftColor),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(width: 10),
        Text('$bLabel $bPct%',
            style: TextStyle(fontWeight: FontWeight.w600, color: scheme.onSurfaceVariant, fontSize: 13)),
      ],
    );
  }
}

class _MetricsBody extends StatelessWidget {
  const _MetricsBody({required this.metrics, required this.scheme});
  final MatchMetrics metrics;
  final ColorScheme scheme;

  @override
  Widget build(BuildContext context) {
    final rows = <(String, String?, String?)>[
      ('xG', _fmt(metrics.homeXg), _fmt(metrics.awayXg)),
      ('Córners', _fmt(metrics.homeCornersAvg), _fmt(metrics.awayCornersAvg)),
      ('Tarjetas', _fmt(metrics.homeYellowCardsAvg), _fmt(metrics.awayYellowCardsAvg)),
    ];
    return Column(
      children: rows.where((r) => r.$2 != null || r.$3 != null).map((r) {
        return Padding(
          padding: const EdgeInsets.symmetric(vertical: 3),
          child: Row(
            children: [
              Expanded(
                child: Text(r.$2 ?? '-',
                    textAlign: TextAlign.start,
                    style: const TextStyle(fontWeight: FontWeight.w600)),
              ),
              Text(r.$1, style: TextStyle(color: scheme.onSurfaceVariant, fontSize: 12)),
              Expanded(
                child: Text(r.$3 ?? '-',
                    textAlign: TextAlign.end,
                    style: const TextStyle(fontWeight: FontWeight.w600)),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }

  static String? _fmt(double? v) => v == null ? null : v.toStringAsFixed(1);
}

class _NoMarket extends StatelessWidget {
  const _NoMarket(this.text);
  final String text;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Text(
        text,
        style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey),
      ),
    );
  }
}

/// Detalle en Modal Bottom Sheet con botón IA prominente.
class FixtureDetailSheet extends StatelessWidget {
  const FixtureDetailSheet({super.key, required this.fixture});

  final FixtureResponse fixture;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      decoration: BoxDecoration(
        color: scheme.surface,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
      ),
      child: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
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
              const SizedBox(height: 16),
              // Header teams
              Row(
                children: [
                  _TeamColumn(team: fixture.homeTeam),
                  Expanded(
                    child: Text(
                      'vs',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            color: scheme.onSurfaceVariant,
                            fontWeight: FontWeight.w700,
                          ),
                      textAlign: TextAlign.center,
                    ),
                  ),
                  _TeamColumn(team: fixture.awayTeam),
                ],
              ),
              const SizedBox(height: 8),
              Text(
                '${fixture.competition.name} · ${DateFormat('EEE dd MMM HH:mm', 'es').format(fixture.kickoffTime.toLocal())}',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
              ),
              if (fixture.venue != null) ...[
                const SizedBox(height: 4),
                Text(
                  fixture.venue!,
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
                ),
              ],
              const SizedBox(height: 20),
              if (fixture.markets?.w1x2 != null) ...[
                _W1X2Body(w1x2: fixture.markets!.w1x2!, scheme: scheme),
                const SizedBox(height: 20),
              ],
              if (fixture.markets?.goals != null) ...[
                _GoalsBody(goals: fixture.markets!.goals!, scheme: scheme),
                const SizedBox(height: 20),
              ],
              if (fixture.markets?.metrics != null) ...[
                Text('Métricas', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
                const SizedBox(height: 8),
                _MetricsBody(metrics: fixture.markets!.metrics!, scheme: scheme),
                const SizedBox(height: 20),
              ],
              _AiButton(fixture: fixture),
            ],
          ),
        ),
      ),
    );
  }
}

class _AiButton extends ConsumerWidget {
  const _AiButton({required this.fixture});

  final FixtureResponse fixture;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final w = fixture.markets?.w1x2;
    if (w == null) {
      return const SizedBox.shrink();
    }
    return FilledButton.icon(
      icon: const Icon(Icons.auto_awesome),
      label: const Text('✨ Analizar con IA (NVIDIA)'),
      style: FilledButton.styleFrom(
        padding: const EdgeInsets.symmetric(vertical: 14),
        backgroundColor: Theme.of(context).colorScheme.tertiaryContainer,
        foregroundColor: Theme.of(context).colorScheme.onTertiaryContainer,
      ),
      onPressed: () async {
        final client = ref.read(apiClientProvider);
        final explanation = await _fetchExplanation(client, fixture, w);
        if (context.mounted) {
          _showExplanationSheet(context, explanation);
        }
      },
    );
  }

  Future<String> _fetchExplanation(
    ApiClient client,
    FixtureResponse fixture,
    W1X2Market w,
  ) async {
    try {
      final resp = await client.dio.post<Map<String, dynamic>>(
        '/api/v1/explain',
        data: {
          'fixture_id': fixture.id,
          'prob_home': w.home,
          'prob_draw': w.draw,
          'prob_away': w.away,
          'home_team': fixture.homeTeam.name,
          'away_team': fixture.awayTeam.name,
          'metrics': fixture.markets?.metrics?.toJson(),
        },
      );
      final data = resp.data;
      return (data != null ? data['explanation'] as String? : null) ?? 'Sin explicación.';
    } catch (e) {
      return 'No se pudo obtener la explicación IA.';
    }
  }

  void _showExplanationSheet(BuildContext context, String text) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => Container(
        constraints: BoxConstraints(maxHeight: MediaQuery.of(ctx).size.height * 0.7),
        decoration: BoxDecoration(
          color: Theme.of(ctx).colorScheme.surface,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
        ),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(Icons.auto_awesome, color: Colors.amber),
                  const SizedBox(width: 8),
                  Text('Análisis IA',
                      style: Theme.of(ctx).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
                  const Spacer(),
                  IconButton(icon: const Icon(Icons.close), onPressed: () => Navigator.pop(ctx)),
                ],
              ),
              const SizedBox(height: 12),
              Flexible(
                child: SingleChildScrollView(
                  child: Text(text, style: Theme.of(ctx).textTheme.bodyMedium),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}