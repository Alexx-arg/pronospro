import 'package:football_prediction_app/features/prediction/data/models/prediction_response.dart';

/// Estados posibles de la pantalla de predicción — Sprint 7.2
/// D12: separación estricta, sin lógica de red aquí.
sealed class PredictionState {
  const PredictionState();
}

class PredictionInitial extends PredictionState {
  const PredictionInitial();
}

class PredictionLoading extends PredictionState {
  const PredictionLoading();
}

class PredictionSuccess extends PredictionState {
  const PredictionSuccess(this.response);
  final PredictionResponse response;
}

class PredictionError extends PredictionState {
  const PredictionError(this.message);
  final String message;
}
