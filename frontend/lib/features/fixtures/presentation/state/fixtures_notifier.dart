import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:football_prediction_app/core/network/api_client.dart';
import 'package:football_prediction_app/core/network/exceptions.dart';
import 'package:football_prediction_app/features/fixtures/fixtures.dart';

final fixturesRepositoryProvider = Provider<FixturesRepository>((ref) {
  final client = ref.read(apiClientProvider);
  return FixturesRepositoryImpl(apiClient: client);
});

/// Selected day (stored as a day-index offset: -1 ayer, 0 hoy, +n).
final selectedDateProvider = StateProvider<int>((ref) => 0);

final fixturesNotifierProvider =
    StateNotifierProvider<FixturesNotifier, FixturesState>((ref) {
  final repo = ref.read(fixturesRepositoryProvider);
  return FixturesNotifier(repo);
});

class FixturesNotifier extends StateNotifier<FixturesState> {
  FixturesNotifier(this._repository) : super(const FixturesInitial());

  final FixturesRepository _repository;

  Future<void> loadForDate(
    DateTime day, {
    int? competitionId,
    bool refresh = false,
  }) async {
    state = const FixturesLoading();
    try {
      final result = await _repository.getUpcoming(
        date: day,
        competitionId: competitionId,
        limit: 100,
        offset: 0,
        includeMetrics: true,
        includePrediction: true,
      );
      state = FixturesSuccess(result, true);
    } on UnauthorizedException {
      state = const FixturesError('Error de credenciales');
    } on ServerException catch (e) {
      state = FixturesError('Error del servidor: ${e.message}');
    } catch (e) {
      state = FixturesError('Error inesperado: $e');
    }
  }

  void reset() => state = const FixturesInitial();
}

class FixtureDetailNotifier extends StateNotifier<FixtureDetailState> {
  FixtureDetailNotifier(this._repository) : super(const FixtureDetailInitial());

  final FixturesRepository _repository;

  Future<void> loadFixture(int fixtureId) async {
    state = const FixtureDetailLoading();
    try {
      final fixture = await _repository.getFixture(
        fixtureId,
        includeMetrics: true,
        includePrediction: true,
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