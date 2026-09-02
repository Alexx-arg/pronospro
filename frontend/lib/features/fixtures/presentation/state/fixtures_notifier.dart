import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:football_prediction_app/core/network/api_client.dart';
import 'package:football_prediction_app/core/network/exceptions.dart';
import 'package:football_prediction_app/features/fixtures/fixtures.dart';

final fixturesRepositoryProvider = Provider<FixturesRepository>((ref) {
  final client = ref.read(apiClientProvider);
  return FixturesRepositoryImpl(apiClient: client);
});

final fixturesNotifierProvider =
    StateNotifierProvider<FixturesNotifier, FixturesState>((ref) {
  final repo = ref.read(fixturesRepositoryProvider);
  return FixturesNotifier(repo);
});

final fixtureDetailProvider =
    StateNotifierProvider<FixtureDetailNotifier, FixtureDetailState>((ref) {
  final repo = ref.read(fixturesRepositoryProvider);
  return FixtureDetailNotifier(repo);
});

class FixturesNotifier extends StateNotifier<FixturesState> {
  FixturesNotifier(this._repository) : super(const FixturesInitial());

  final FixturesRepository _repository;
  int _offset = 0;
  final int _limit = 20;
  bool _hasReachedMax = false;

  Future<void> loadUpcoming({
    int? competitionId,
    int? seasonId,
    int? teamId,
    DateTime? from,
    DateTime? to,
    bool includeMetrics = false,
    bool includePrediction = false,
    bool refresh = false,
  }) async {
    if (refresh) {
      _offset = 0;
      _hasReachedMax = false;
      state = const FixturesLoading();
    } else if (state is FixturesLoading || _hasReachedMax) {
      return;
    }

    try {
      final result = await _repository.getUpcoming(
        competitionId: competitionId,
        seasonId: seasonId,
        teamId: teamId,
        from: from,
        to: to,
        limit: _limit,
        offset: _offset,
        includeMetrics: includeMetrics,
        includePrediction: includePrediction,
      );

      if (refresh || _offset == 0) {
        state = FixturesSuccess(
          result,
          result.items.length < _limit || result.items.isEmpty,
        );
      } else if (state is FixturesSuccess) {
        final current = state as FixturesSuccess;
        final newItems = [...current.fixtures.items, ...result.items];
        state = FixturesSuccess(
          PaginatedFixtures(
            items: newItems,
            limit: result.limit,
            offset: result.offset,
            total: result.total,
          ),
          newItems.length >= result.total || result.items.length < _limit,
        );
      }
      _offset += result.items.length;
      _hasReachedMax = result.items.length < _limit;
    } on FixtureNotFoundException {
      state = const FixturesError('No se encontraron partidos');
    } on UnauthorizedException {
      state = const FixturesError('Error de credenciales');
    } on ServerException catch (e) {
      state = FixturesError('Error del servidor: ${e.message}');
    } catch (e) {
      state = FixturesError('Error inesperado: $e');
    }
  }

  Future<void> loadMore({
    int? competitionId,
    int? seasonId,
    int? teamId,
    DateTime? from,
    DateTime? to,
    bool includeMetrics = false,
    bool includePrediction = false,
  }) async {
    if (state is FixturesLoading || _hasReachedMax) return;
    await loadUpcoming(
      competitionId: competitionId,
      seasonId: seasonId,
      teamId: teamId,
      from: from,
      to: to,
      includeMetrics: includeMetrics,
      includePrediction: includePrediction,
      refresh: false,
    );
  }

  void reset() {
    _offset = 0;
    _hasReachedMax = false;
    state = const FixturesInitial();
  }
}

class FixtureDetailNotifier extends StateNotifier<FixtureDetailState> {
  FixtureDetailNotifier(this._repository) : super(const FixtureDetailInitial());

  final FixturesRepository _repository;

  Future<void> loadFixture(
    int fixtureId, {
    bool includeMetrics = true,
    bool includePrediction = true,
  }) async {
    state = const FixtureDetailLoading();
    try {
      final fixture = await _repository.getFixture(
        fixtureId,
        includeMetrics: includeMetrics,
        includePrediction: includePrediction,
      );
      state = FixtureDetailSuccess(fixture);
    } on FixtureNotFoundException {
      state = const FixtureDetailError('Partido no encontrado');
    } on UnauthorizedException {
      state = const FixtureDetailError('Error de credenciales');
    } on ServerException catch (e) {
      state = FixtureDetailError('Error del servidor: ${e.message}');
    } catch (e) {
      state = FixtureDetailError('Error inesperado: $e');
    }
  }

  void reset() => state = const FixtureDetailInitial();
}