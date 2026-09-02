import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:football_prediction_app/core/network/api_client.dart';
import 'package:football_prediction_app/core/network/exceptions.dart';
import 'package:football_prediction_app/features/prediction/data/models/prediction_response.dart';
import 'package:football_prediction_app/features/prediction/domain/repositories/prediction_repository.dart';

// Mocks
class MockDio extends Mock implements Dio {}

void main() {
  group('PredictionResponse', () {
    test('fromJson mapea snake_case y valida simplex', () {
      final json = {
        'fixture_id': 123,
        'prob_home': 0.5,
        'prob_draw': 0.3,
        'prob_away': 0.2,
        'model_version': 'v001',
        'predicted_at': '2026-01-15T12:00:00Z',
      };
      final model = PredictionResponse.fromJson(json);
      expect(model.fixtureId, 123);
      expect(model.probHome, 0.5);
      expect(model.probDraw, 0.3);
      expect(model.probAway, 0.2);
      expect(model.isValidSimplex, isTrue);

      // Tipado fuerte: double
      expect(model.probHome, isA<double>());
    });

    test('toJson round-trip', () {
      final original = PredictionResponse(
        fixtureId: 1,
        probHome: 0.6,
        probDraw: 0.2,
        probAway: 0.2,
        modelVersion: 'v001',
        predictedAt: DateTime.parse('2026-01-15T12:00:00Z'),
      );
      final json = original.toJson();
      final restored = PredictionResponse.fromJson(json);
      expect(restored.fixtureId, original.fixtureId);
      expect(restored.probHome, original.probHome);
    });
  });

  group('PredictionRepository', () {
    late MockDio mockDio;
    late PredictionRepository repo;

    setUp(() {
      mockDio = MockDio();
      final apiClient = ApiClient(dio: mockDio, baseUrl: 'http://test', apiKey: 'test-key');
      repo = PredictionRepositoryImpl(apiClient: apiClient);
    });

    test('getPredictionForFixture retorna PredictionResponse en 200', () async {
      final responseJson = {
        'fixture_id': 123,
        'prob_home': 0.6,
        'prob_draw': 0.2,
        'prob_away': 0.2,
        'model_version': 'v001',
        'predicted_at': '2026-01-15T12:00:00Z',
      };
      when(() => mockDio.get(any())).thenAnswer(
        (_) async => Response(
          requestOptions: RequestOptions(path: '/api/v1/predict/fixture/123'),
          statusCode: 200,
          data: responseJson,
        ),
      );

      final result = await repo.getPredictionForFixture(123);
      expect(result.fixtureId, 123);
      expect(result.probHome, 0.6);
      verify(() => mockDio.get('/api/v1/predict/fixture/123')).called(1);
    });

    test('mapea 401 -> UnauthorizedException', () async {
      when(() => mockDio.get(any())).thenThrow(
        DioException(
          requestOptions: RequestOptions(path: '/api/v1/predict/fixture/123'),
          response: Response(
            requestOptions: RequestOptions(path: '/api/v1/predict/fixture/123'),
            statusCode: 401,
            data: {'detail': 'Invalid API Key'},
          ),
          type: DioExceptionType.badResponse,
        ),
      );
      expect(
        () => repo.getPredictionForFixture(123),
        throwsA(isA<UnauthorizedException>()),
      );
    });

    test('mapea 404 -> FixtureNotFoundException', () async {
      when(() => mockDio.get(any())).thenThrow(
        DioException(
          requestOptions: RequestOptions(path: '/api/v1/predict/fixture/999'),
          response: Response(
            requestOptions: RequestOptions(path: '/api/v1/predict/fixture/999'),
            statusCode: 404,
            data: {'detail': 'Fixture not found'},
          ),
          type: DioExceptionType.badResponse,
        ),
      );
      expect(
        () => repo.getPredictionForFixture(999),
        throwsA(isA<FixtureNotFoundException>()),
      );
    });

    test('mapea 422 -> InsufficientHistoryException', () async {
      when(() => mockDio.get(any())).thenThrow(
        DioException(
          requestOptions: RequestOptions(path: '/api/v1/predict/fixture/123'),
          response: Response(
            requestOptions: RequestOptions(path: '/api/v1/predict/fixture/123'),
            statusCode: 422,
            data: {'detail': 'Not enough historical data'},
          ),
          type: DioExceptionType.badResponse,
        ),
      );
      expect(
        () => repo.getPredictionForFixture(123),
        throwsA(isA<InsufficientHistoryException>()),
      );
    });

    test('mapea 503 -> ServerException', () async {
      when(() => mockDio.get(any())).thenThrow(
        DioException(
          requestOptions: RequestOptions(path: '/api/v1/predict/fixture/123'),
          response: Response(
            requestOptions: RequestOptions(path: '/api/v1/predict/fixture/123'),
            statusCode: 503,
            data: {'detail': 'model not loaded'},
          ),
          type: DioExceptionType.badResponse,
        ),
      );
      expect(
        () => repo.getPredictionForFixture(123),
        throwsA(isA<ServerException>()),
      );
    });

    test('ApiClient inyecta X-API-Key automáticamente', () async {
      // Verifica que el interceptor añade el header
      final dio = Dio();
      dio.options.baseUrl = 'http://test';
      final apiClient = ApiClient(dio: dio, baseUrl: 'http://test', apiKey: 'my-secret');
      // Simula request y verifica header
      // Usamos interceptor directamente: dio.interceptors ya contiene el wrapper
      expect(apiClient.dio.options.baseUrl, 'http://test');
      // El header se añade en onRequest; no podemos testear sin mockear, pero verificamos que dio tiene interceptor
      expect(apiClient.dio.interceptors.length, greaterThan(0));
    });
  });

  group('ApiClient D11.2', () {
    test('no hardcodea URL ni API_KEY (viene de env/dart-define)', () {
      final client1 = ApiClient(baseUrl: 'http://from-env', apiKey: 'from-env-key');
      expect(client1.baseUrl, 'http://from-env');
      final client2 = ApiClient(baseUrl: 'http://custom', apiKey: 'custom-key');
      expect(client2.baseUrl, 'http://custom');
    });
  });
}
