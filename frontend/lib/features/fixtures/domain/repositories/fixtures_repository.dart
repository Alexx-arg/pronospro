import 'package:dio/dio.dart';

import 'package:football_prediction_app/core/network/api_client.dart';
import 'package:football_prediction_app/core/network/exceptions.dart';
import 'package:football_prediction_app/features/fixtures/fixtures.dart';

abstract class FixturesRepository {
  Future<PaginatedFixtures> getUpcoming({
    int? competitionId,
    int? seasonId,
    int? teamId,
    DateTime? from,
    DateTime? to,
    int limit = 20,
    int offset = 0,
    bool includeMetrics = false,
    bool includePrediction = false,
  });

  Future<FixtureResponse> getFixture(
    int fixtureId, {
    bool includeMetrics = true,
    bool includePrediction = true,
  });
}

class FixturesRepositoryImpl implements FixturesRepository {
  FixturesRepositoryImpl({required this.apiClient});

  final ApiClient apiClient;

  @override
  Future<PaginatedFixtures> getUpcoming({
    int? competitionId,
    int? seasonId,
    int? teamId,
    DateTime? from,
    DateTime? to,
    int limit = 20,
    int offset = 0,
    bool includeMetrics = false,
    bool includePrediction = false,
  }) async {
    try {
      final queryParams = <String, dynamic>{
        'limit': limit,
        'offset': offset,
        'include_metrics': includeMetrics,
        'include_prediction': includePrediction,
      };
      if (competitionId != null) queryParams['competition_id'] = competitionId;
      if (seasonId != null) queryParams['season_id'] = seasonId;
      if (teamId != null) queryParams['team_id'] = teamId;
      if (from != null) queryParams['from'] = from.toIso8601String();
      if (to != null) queryParams['to'] = to.toIso8601String();

      final response = await apiClient.dio.get(
        '/api/v1/fixtures/upcoming',
        queryParameters: queryParams,
      );

      if (response.statusCode == 200) {
        return PaginatedFixtures.fromJson(response.data as Map<String, dynamic>);
      }

      throw _mapStatusCode(response.statusCode, response.data);
    } on DioException catch (e) {
      if (e.response == null) {
        // Network error — no response received
        throw ServerException('Sin conexión con el backend (${apiClient.baseUrl}). Verifica que está corriendo y usa "usesCleartextTraffic=true" para HTTP');
      }
      throw _mapStatusCode(e.response?.statusCode, e.response?.data);
    } catch (e) {
      throw ServerException('Error inesperado: $e — backend: ${apiClient.baseUrl}');
    }
  }

  @override
  Future<FixtureResponse> getFixture(
    int fixtureId, {
    bool includeMetrics = true,
    bool includePrediction = true,
  }) async {
    try {
      final response = await apiClient.dio.get(
        '/api/v1/fixtures/$fixtureId',
        queryParameters: {
          'include_metrics': includeMetrics,
          'include_prediction': includePrediction,
        },
      );

      if (response.statusCode == 200) {
        return FixtureResponse.fromJson(response.data as Map<String, dynamic>);
      }

      throw _mapStatusCode(response.statusCode, response.data);
    } on DioException catch (e) {
      if (e.response == null) {
        // Network error — no response received
        throw ServerException('Sin conexión con el backend (${apiClient.baseUrl}). Verifica que está corriendo y usa "usesCleartextTraffic=true" para HTTP');
      }
      throw _mapStatusCode(e.response?.statusCode, e.response?.data);
    } catch (e) {
      throw ServerException('Error inesperado: $e — backend: ${apiClient.baseUrl}');
    }
  }

  Exception _mapStatusCode(int? statusCode, dynamic data) {
    final message = (data is Map && data['detail'] is String)
        ? data['detail'] as String
        : 'Unknown error';
    switch (statusCode) {
      case 401:
        return UnauthorizedException(message);
      case 404:
        return FixtureNotFoundException(message);
      case 500:
      case 503:
        return ServerException(message);
      default:
        return ServerException('Unexpected status $statusCode: $message');
    }
  }
}