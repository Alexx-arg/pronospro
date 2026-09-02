/// Modelo de respuesta del endpoint GET /api/v1/predict/fixture/{id}
/// Mapea MatchPredictionResponse de FastAPI (snake_case) a Dart (camelCase).
/// Tipado fuerte D11.3: double no nulo para probabilidades.
class PredictionResponse {
  const PredictionResponse({
    required this.fixtureId,
    required this.probHome,
    required this.probDraw,
    required this.probAway,
    required this.modelVersion,
    required this.predictedAt,
  });

  final int fixtureId;
  final double probHome;
  final double probDraw;
  final double probAway;
  final String modelVersion;
  final DateTime predictedAt;

  /// Validación simplex: suma ≈1 (tolerancia 1e-6) — espejo de validación backend.
  bool get isValidSimplex {
    final sum = probHome + probDraw + probAway;
    return (sum - 1.0).abs() < 1e-6 &&
        probHome >= 0 &&
        probHome <= 1 &&
        probDraw >= 0 &&
        probDraw <= 1 &&
        probAway >= 0 &&
        probAway <= 1;
  }

  factory PredictionResponse.fromJson(Map<String, dynamic> json) {
    return PredictionResponse(
      fixtureId: json['fixture_id'] as int,
      probHome: (json['prob_home'] as num).toDouble(),
      probDraw: (json['prob_draw'] as num).toDouble(),
      probAway: (json['prob_away'] as num).toDouble(),
      modelVersion: json['model_version'] as String,
      predictedAt: DateTime.parse(json['predicted_at'] as String),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'fixture_id': fixtureId,
      'prob_home': probHome,
      'prob_draw': probDraw,
      'prob_away': probAway,
      'model_version': modelVersion,
      'predicted_at': predictedAt.toIso8601String(),
    };
  }
}
