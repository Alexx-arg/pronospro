import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import 'package:football_prediction_app/core/network/api_client.dart';
import 'package:football_prediction_app/features/fixtures/fixtures.dart';

class FixturesScreen extends ConsumerStatefulWidget {
  const FixturesScreen({super.key});

  @override
  ConsumerState<FixturesScreen> createState() => _FixturesScreenState();
}

class _FixturesScreenState extends ConsumerState<FixturesScreen> {
  final ScrollController _scrollController = ScrollController();
  int? _selectedLeagueId;
  final Map<int, String> _leagueNames = const {
    39: 'Premier League',
    140: 'La Liga',
    135: 'Serie A',
    78: 'Bundesliga',
    61: 'Ligue 1',
  };

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(_onScroll);
    _loadFixtures();
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  void _onScroll() {
    if (_scrollController.position.pixels >=
        _scrollController.position.maxScrollExtent - 200) {
      _loadMore();
    }
  }

  void _loadFixtures({bool refresh = false}) {
    ref.read(fixturesNotifierProvider.notifier).loadUpcoming(
      competitionId: _selectedLeagueId,
      from: DateTime.now().subtract(const Duration(days: 1)),
      to: DateTime.now().add(const Duration(days: 7)),
      includeMetrics: true,
      includePrediction: true,
      refresh: refresh,
    );
  }

  void _loadMore() {
    ref.read(fixturesNotifierProvider.notifier).loadMore(
      competitionId: _selectedLeagueId,
      from: DateTime.now().subtract(const Duration(days: 1)),
      to: DateTime.now().add(const Duration(days: 7)),
      includeMetrics: true,
      includePrediction: true,
    );
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(fixturesNotifierProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Próximos Partidos'),
        actions: [
          PopupMenuButton<int?>(
            initialValue: _selectedLeagueId,
            onSelected: (value) {
              setState(() {
                _selectedLeagueId = value;
              });
              _loadFixtures(refresh: true);
            },
            itemBuilder: (context) => [
              const PopupMenuItem<int?>(
                value: null,
                child: Text('Todas las ligas'),
              ),
              ..._leagueNames.entries.map(
                (e) => PopupMenuItem<int?>(
                  value: e.key,
                  child: Text(e.value),
                ),
              ),
            ],
            icon: const Icon(Icons.filter_list),
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: () async => _loadFixtures(refresh: true),
        child: _buildBody(state),
      ),
    );
  }

  Widget _buildBody(FixturesState state) {
    return switch (state) {
      FixturesInitial() => const Center(
          child: Text('Cargando partidos...'),
        ),
      FixturesLoading() => const Center(child: CircularProgressIndicator()),
      FixturesSuccess(:final fixtures, :final hasReachedMax) => fixtures.items.isEmpty
          ? const Center(
              child: Text('No hay partidos programados para esta liga'),
            )
          : ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.all(12),
              itemCount: fixtures.items.length + (hasReachedMax ? 0 : 1),
              itemBuilder: (context, index) {
                if (index >= fixtures.items.length) {
                  return const Padding(
                    padding: EdgeInsets.all(16),
                    child: Center(child: CircularProgressIndicator()),
                  );
                }
                final fixture = fixtures.items[index];
                return FixtureCard(fixture: fixture);
              },
            ),
      FixturesError(:final message) => Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(message, style: const TextStyle(color: Colors.red)),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: _loadFixtures,
                child: const Text('Reintentar'),
              ),
            ],
          ),
        ),
    };
  }
}