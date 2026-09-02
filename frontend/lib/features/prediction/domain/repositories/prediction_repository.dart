import 'package:dio/dio.dart';

import 'package:football_prediction_app/core/network/api_client.dart';
import 'package:football_prediction_app/core/network/exceptions.dart';
import 'package:football_prediction_app/features/prediction/data/models/prediction_response.dart';

abstract class PredictionRepository {
  Future<PredictionResponse> getPredictionForFixture(int fixtureId);
}

class PredictionRepositoryImpl implements PredictionRepository {
  PredictionRepositoryImpl({required this.apiClient});

  final ApiClient apiClient;

  @override
  Future<PredictionResponse> getPredictionForFixture(int fixtureId) async {
    try {
      final response = await apiClient.dio.get(
        '/api/v1/predict/fixture/$fixtureId',
      );

      if (response.statusCode == 200) {
        final data = response.data as Map<String, dynamic>;
        return PredictionResponse.fromJson(data);
      }

      // Dio normalmente lanza DioException para status !=2xx, pero por si acaso
      throw _mapStatusCode(response.statusCode, response.data);
    } on DioException catch (e) {
      if (e.response == null) {
        final baseUrl = apiClient.dio.options.baseUrl;
        throw ServerException('Sin conexión al backend ($baseUrl). Verifica que está en ejecución.');
      }
      final status = e.response?.statusCode;
      final data = e.response?.data;
      throw _mapStatusCode(status, data);
    } catch (e) {
      final baseUrl = apiClient.dio.options.baseUrl;
      throw ServerException('Error inesperado: $e — backend: $baseUrl');
    }
  }

  Exception _mapStatusCode(int? statusCode, dynamic data) {
    final message = (data is Map && data['detail'] is String) ? data['detail'] as String : 'Unknown error';
    switch (statusCode) {
      case 401:
        return UnauthorizedException(message);
      case 404:
        return FixtureNotFoundException(message);
      case 422:
        return InsufficientHistoryException(message);
      case 500:
      case 503:
        return ServerException(message);
      default:
        return ServerException('Unexpected status $statusCode: $message');
    }
  }
}
