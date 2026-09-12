import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:football_prediction_app/features/fixtures/fixtures.dart';

class FixturesScreen extends ConsumerStatefulWidget {
  const FixturesScreen({super.key});

  @override
  ConsumerState<FixturesScreen> createState() => _FixturesScreenState();
}

class _FixturesScreenState extends ConsumerState<FixturesScreen> {
  int _offset = 0; // -1 ayer, 0 hoy, +n

  @override
  void initState() {
    super.initState();
    _load();
  }

  DateTime _selectedDay() => DateTime.now().add(Duration(days: _offset));

  void _load() {
    ref.read(fixturesNotifierProvider.notifier).loadForDate(_selectedDay());
    ref.read(selectedDateProvider.notifier).state = _offset;
  }

  void _setDay(int offset) {
    setState(() => _offset = offset);
    _load();
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(fixturesNotifierProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('PronosPro'),
        centerTitle: false,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _load,
          ),
        ],
      ),
      body: Column(
        children: [
          DateSelectorBar(
            offset: _offset,
            onSelect: _setDay,
          ),
          Expanded(child: _buildBody(state)),
        ],
      ),
    );
  }

  Widget _buildBody(FixturesState state) {
    return switch (state) {
      FixturesInitial() => const Center(child: CircularProgressIndicator()),
      FixturesLoading() => const Center(child: CircularProgressIndicator()),
      FixturesSuccess(:final fixtures) => fixtures.items.isEmpty
          ? const _EmptyState()
          : _buildGroupedList(fixtures),
      FixturesError(:final message) => _ErrorState(message: message, onRetry: _load),
    };
  }

  Widget _buildGroupedList(PaginatedFixtures paginated) {
    final grouped = paginated.groupedByCompetition();
    final compNames = {for (final f in paginated.items) f.competition.id: f.competition};

    final items = <Widget>[];
    grouped.forEach((compId, fixtures) {
      fixtures.sort((a, b) => a.kickoffTime.compareTo(b.kickoffTime));
      final comp = compNames[compId]!;
      items.add(_LeagueHeader(competition: comp));
      for (final f in fixtures) {
        items.add(FixtureCard(fixture: f));
      }
    });

    return RefreshIndicator(
      onRefresh: () async => _load(),
      child: ListView(
        padding: const EdgeInsets.fromLTRB(12, 4, 12, 24),
        children: items,
      ),
    );
  }
}

class _LeagueHeader extends StatelessWidget {
  const _LeagueHeader({required this.competition});

  final CompetitionInfo competition;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.fromLTRB(4, 14, 4, 6),
      child: Row(
        children: [
          if (competition.logo != null)
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: Image.network(
                competition.logo!,
                width: 22,
                height: 22,
                errorBuilder: (_, __, ___) => const Icon(Icons.emoji_events, size: 22),
              ),
            )
          else
            const Icon(Icons.emoji_events, size: 22, color: Colors.amber),
          const SizedBox(width: 8),
          Text(
            competition.name,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w700,
                  color: scheme.primary,
                ),
          ),
        ],
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.sports_soccer, size: 56, color: Colors.grey),
          const SizedBox(height: 12),
          Text(
            'No hay partidos para esta fecha',
            style: Theme.of(context).textTheme.bodyLarge,
          ),
          const SizedBox(height: 4),
          Text(
            'Deslizá para recargar o seleccioná otro día',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey),
          ),
        ],
      ),
    );
  }
}

class _ErrorState extends StatelessWidget {
  const _ErrorState({required this.message, required this.onRetry});

  final String message;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.error_outline, size: 56, color: Colors.redAccent),
          const SizedBox(height: 12),
          Text(message, textAlign: TextAlign.center, style: const TextStyle(color: Colors.red)),
          const SizedBox(height: 16),
          ElevatedButton(onPressed: onRetry, child: const Text('Reintentar')),
        ],
      ),
    );
  }
}