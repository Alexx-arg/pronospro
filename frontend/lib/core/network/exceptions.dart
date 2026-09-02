/// Excepciones de dominio mapeadas desde HTTP — Sprint 7.1

class UnauthorizedException implements Exception {
  const UnauthorizedException([this.message = 'Unauthorized']);
  final String message;
  @override
  String toString() => 'UnauthorizedException: $message';
}

class FixtureNotFoundException implements Exception {
  const FixtureNotFoundException([this.message = 'Fixture not found']);
  final String message;
  @override
  String toString() => 'FixtureNotFoundException: $message';
}

class InsufficientHistoryException implements Exception {
  const InsufficientHistoryException([this.message = 'Not enough historical data']);
  final String message;
  @override
  String toString() => 'InsufficientHistoryException: $message';
}

class ServerException implements Exception {
  const ServerException([this.message = 'Server error']);
  final String message;
  @override
  String toString() => 'ServerException: $message';
}
