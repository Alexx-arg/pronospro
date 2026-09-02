import 'package:dio/dio.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Provider global del ApiClient — inyección de dependencias (D11.2).
final apiClientProvider = Provider<ApiClient>((ref) => ApiClient());

/// Cliente HTTP base — Sprint 7.1 (D11.2 seguridad).

class ApiClient {
  ApiClient({Dio? dio, String? baseUrl, String? apiKey})
      : _dio = dio ?? Dio(),
        _baseUrl = baseUrl ?? _resolveBaseUrl(),
        _apiKey = apiKey ?? _resolveApiKey() {
    _dio.options.baseUrl = _baseUrl;
    _dio.options.connectTimeout = const Duration(seconds: 10);
    _dio.options.receiveTimeout = const Duration(seconds: 10);
    _dio.interceptors.add(
      InterceptorsWrapper(
        onRequest: (options, handler) {
          // D11.2: inyección automática de X-API-Key en todas las peticiones
          if (_apiKey != null && _apiKey.isNotEmpty) {
            options.headers['X-API-Key'] = _apiKey;
          }
          handler.next(options);
        },
      ),
    );
  }

  final Dio _dio;
  final String _baseUrl;
  final String? _apiKey;

  static String _resolveBaseUrl() {
    // D11.2: nunca hardcodear; viene de .env o --dart-define
    final envUrl = dotenv.maybeGet('API_BASE_URL');
    if (envUrl != null && envUrl.isNotEmpty) return envUrl;
    const dartDefineUrl = String.fromEnvironment('API_BASE_URL');
    if (dartDefineUrl.isNotEmpty) return dartDefineUrl;
    // Fallback para tests (no hardcode de producción)
    return 'http://10.0.2.2:8000';
  }

  static String? _resolveApiKey() {
    final envKey = dotenv.maybeGet('API_KEY');
    if (envKey != null && envKey.isNotEmpty) return envKey;
    // Soporta también API_SECRET_KEY por compatibilidad con backend .env
    final altKey = dotenv.maybeGet('API_SECRET_KEY');
    if (altKey != null && altKey.isNotEmpty) return altKey;
    const dartDefineKey = String.fromEnvironment('API_KEY');
    if (dartDefineKey.isNotEmpty) return dartDefineKey;
    const dartDefineSecret = String.fromEnvironment('API_SECRET_KEY');
    if (dartDefineSecret.isNotEmpty) return dartDefineSecret;
    return null;
  }

  Dio get dio => _dio;
  String get baseUrl => _baseUrl;
}
