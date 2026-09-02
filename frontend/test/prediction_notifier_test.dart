import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:football_prediction_app/core/network/exceptions.dart';
import 'package:football_prediction_app/features/prediction/data/models/prediction_response.dart';
import 'package:football_prediction_app/features/prediction/domain/repositories/prediction_repository.dart';
import 'package:football_prediction_app/features/prediction/presentation/state/prediction_notifier.dart';
import 'package:football_prediction_app/features/prediction/presentation/state/prediction_state.dart';

class MockPredictionRepository extends Mock implements PredictionRepository {}

PredictionResponse _fakeResponse() => PredictionResponse(
      fixtureId: 123,
      probHome: 0.5,
      probDraw: 0.3,
      probAway: 0.2,
      modelVersion: 'v001',
      predictedAt: DateTime.parse('2026-01-15T12:00:00Z'),
    );

void main() {
  group('PredictionNotifier', () {
    late MockPredictionRepository mockRepo;
    late ProviderContainer container;

    setUp(() {
      mockRepo = MockPredictionRepository();
      container = ProviderContainer(
        overrides: [
          predictionRepositoryProvider.overrideWithValue(mockRepo),
        ],
      );
    });

    tearDown(() => container.dispose());

    test('Initial -> Loading -> Success (camino feliz)', () async {
      when(() => mockRepo.getPredictionForFixture(any())).thenAnswer((_) async => _fakeResponse());

      expect(container.read(predictionNotifierProvider), isA<PredictionInitial>());

      final future = container.read(predictionNotifierProvider.notifier).fetchPrediction(123);
      expect(container.read(predictionNotifierProvider), isA<PredictionLoading>());

      await future;

      final state = container.read(predictionNotifierProvider);
      expect(state, isA<PredictionSuccess>());
      expect((state as PredictionSuccess).response.fixtureId, 123);
    });

    test('Initial -> Loading -> Error mapea FixtureNotFound', () async {
      when(() => mockRepo.getPredictionForFixture(any())).thenThrow(const FixtureNotFoundException());

      final notifier = container.read(predictionNotifierProvider.notifier);
      await notifier.fetchPrediction(999);

      final state = container.read(predictionNotifierProvider);
      expect(state, isA<PredictionError>());
      expect((state as PredictionError).message, 'Partido no encontrado');
    });

    test('Mapea InsufficientHistory -> mensaje amigable', () async {
      when(() => mockRepo.getPredictionForFixture(any())).thenThrow(const InsufficientHistoryException());

      await container.read(predictionNotifierProvider.notifier).fetchPrediction(123);

      expect((container.read(predictionNotifierProvider) as PredictionError).message,
          'No hay historial suficiente para este partido');
    });

    test('Mapea Unauthorized -> Error de credenciales', () async {
      when(() => mockRepo.getPredictionForFixture(any())).thenThrow(const UnauthorizedException());

      await container.read(predictionNotifierProvider.notifier).fetchPrediction(123);

      expect((container.read(predictionNotifierProvider) as PredictionError).message, 'Error de credenciales');
    });

    test('Error genérico mapea a mensaje inesperado', () async {
      when(() => mockRepo.getPredictionForFixture(any())).thenThrow(Exception('boom'));

      await container.read(predictionNotifierProvider.notifier).fetchPrediction(123);

      expect((container.read(predictionNotifierProvider) as PredictionError).message, contains('inesperado'));
    });
  });
}
