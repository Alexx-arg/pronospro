import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:football_prediction_app/core/network/api_client.dart';
import 'package:football_prediction_app/core/network/exceptions.dart';
import 'package:football_prediction_app/features/prediction/data/models/prediction_response.dart';
import 'package:football_prediction_app/features/prediction/domain/repositories/prediction_repository.dart';
import 'prediction_state.dart';

// Providers — inyección de dependencias (D12.1: UI no crea Dio)
// apiClientProvider ahora está en core/network/api_client.dart

final predictionRepositoryProvider = Provider<PredictionRepository>((ref) {
  final client = ref.read(apiClientProvider);
  return PredictionRepositoryImpl(apiClient: client);
});

final predictionNotifierProvider =
    StateNotifierProvider<PredictionNotifier, PredictionState>((ref) {
  final repo = ref.read(predictionRepositoryProvider);
  return PredictionNotifier(repo);
});

/// Controlador / Notifier — Sprint 7.2
/// Mapea excepciones de dominio a mensajes amigables para la UI.
class PredictionNotifier extends StateNotifier<PredictionState> {
  PredictionNotifier(this._repository) : super(const PredictionInitial());

  final PredictionRepository _repository;

  Future<void> fetchPrediction(int fixtureId) async {
    state = const PredictionLoading();
    try {
      final result = await _repository.getPredictionForFixture(fixtureId);
      state = PredictionSuccess(result);
    } on FixtureNotFoundException {
      state = const PredictionError('Partido no encontrado');
    } on InsufficientHistoryException {
      state = const PredictionError('No hay historial suficiente para este partido');
    } on UnauthorizedException {
      state = const PredictionError('Error de credenciales');
    } on ServerException catch (e) {
      state = PredictionError('Error del servidor: ${e.message}');
    } catch (e) {
      state = PredictionError('Error inesperado: $e');
    }
  }

  void reset() => state = const PredictionInitial();
}
