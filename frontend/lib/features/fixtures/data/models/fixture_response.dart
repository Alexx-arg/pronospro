/// Fixture model matching backend GET /api/v1/fixtures/* response (PronosPro v2).
/// Includes nested competition, season, teams, and a `markets` block
/// (1X2 / goals over-under+BTTS / quantitative metrics).

class TeamInfo {
  const TeamInfo({
    required this.id,
    required this.externalId,
    required this.name,
    this.shortName,
    this.code,
    this.country,
    this.logo,
    this.venue,
  });

  final int id;
  final int externalId;
  final String name;
  final String? shortName;
  final String? code;
  final String? country;
  final String? logo;
  final String? venue;

  factory TeamInfo.fromJson(Map<String, dynamic> json) {
    return TeamInfo(
      id: json['id'] as int,
      externalId: json['external_id'] as int,
      name: json['name'] as String,
      shortName: json['short_name'] as String?,
      code: json['code'] as String?,
      country: json['country'] as String?,
      logo: json['logo'] as String?,
      venue: json['venue'] as String?,
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'external_id': externalId,
        'name': name,
        'short_name': shortName,
        'code': code,
        'country': country,
        'logo': logo,
        'venue': venue,
      };
}

class CompetitionInfo {
  const CompetitionInfo({
    required this.id,
    required this.externalId,
    required this.name,
    this.logo,
  });

  final int id;
  final int externalId;
  final String name;
  final String? logo;

  factory CompetitionInfo.fromJson(Map<String, dynamic> json) => CompetitionInfo(
        id: json['id'] as int,
        externalId: json['external_id'] as int,
        name: json['name'] as String,
        logo: json['logo'] as String?,
      );

  Map<String, dynamic> toJson() => {
        'id': id,
        'external_id': externalId,
        'name': name,
        'logo': logo,
      };
}

class SeasonInfo {
  const SeasonInfo({
    required this.id,
    required this.externalId,
    required this.year,
  });

  final int id;
  final int externalId;
  final int year;

  factory SeasonInfo.fromJson(Map<String, dynamic> json) => SeasonInfo(
        id: json['id'] as int,
        externalId: json['external_id'] as int,
        year: json['year'] as int,
      );

  Map<String, dynamic> toJson() => {
        'id': id,
        'external_id': externalId,
        'year': year,
      };
}

class MatchMetrics {
  const MatchMetrics({
    this.homeForm,
    this.awayForm,
    this.homeXg,
    this.homeXga,
    this.awayXg,
    this.awayXga,
    this.homeCornersAvg,
    this.awayCornersAvg,
    this.homeYellowCardsAvg,
    this.awayYellowCardsAvg,
    this.homeRedCardsAvg,
    this.awayRedCardsAvg,
    this.homePossessionAvg,
    this.awayPossessionAvg,
  });

  final String? homeForm;
  final String? awayForm;
  final double? homeXg;
  final double? homeXga;
  final double? awayXg;
  final double? awayXga;
  final double? homeCornersAvg;
  final double? awayCornersAvg;
  final double? homeYellowCardsAvg;
  final double? awayYellowCardsAvg;
  final double? homeRedCardsAvg;
  final double? awayRedCardsAvg;
  final double? homePossessionAvg;
  final double? awayPossessionAvg;

  factory MatchMetrics.fromJson(Map<String, dynamic> json) => MatchMetrics(
        homeForm: json['home_form'] as String?,
        awayForm: json['away_form'] as String?,
        homeXg: (json['home_xg'] as num?)?.toDouble(),
        homeXga: (json['home_xga'] as num?)?.toDouble(),
        awayXg: (json['away_xg'] as num?)?.toDouble(),
        awayXga: (json['away_xga'] as num?)?.toDouble(),
        homeCornersAvg: (json['home_corners_avg'] as num?)?.toDouble(),
        awayCornersAvg: (json['away_corners_avg'] as num?)?.toDouble(),
        homeYellowCardsAvg: (json['home_yellow_cards_avg'] as num?)?.toDouble(),
        awayYellowCardsAvg: (json['away_yellow_cards_avg'] as num?)?.toDouble(),
        homeRedCardsAvg: (json['home_red_cards_avg'] as num?)?.toDouble(),
        awayRedCardsAvg: (json['away_red_cards_avg'] as num?)?.toDouble(),
        homePossessionAvg: (json['home_possession_avg'] as num?)?.toDouble(),
        awayPossessionAvg: (json['away_possession_avg'] as num?)?.toDouble(),
      );

  Map<String, dynamic> toJson() => {
        'home_form': homeForm,
        'away_form': awayForm,
        'home_xg': homeXg,
        'home_xga': homeXga,
        'away_xg': awayXg,
        'away_xga': awayXga,
        'home_corners_avg': homeCornersAvg,
        'away_corners_avg': awayCornersAvg,
        'home_yellow_cards_avg': homeYellowCardsAvg,
        'away_yellow_cards_avg': awayYellowCardsAvg,
        'home_red_cards_avg': homeRedCardsAvg,
        'away_red_cards_avg': awayRedCardsAvg,
        'home_possession_avg': homePossessionAvg,
        'away_possession_avg': awayPossessionAvg,
      };
}

class W1X2Market {
  const W1X2Market({
    required this.home,
    required this.draw,
    required this.away,
    this.modelVersion,
  });

  final double home;
  final double draw;
  final double away;
  final String? modelVersion;

  factory W1X2Market.fromJson(Map<String, dynamic> json) => W1X2Market(
        home: (json['home'] as num).toDouble(),
        draw: (json['draw'] as num).toDouble(),
        away: (json['away'] as num).toDouble(),
        modelVersion: json['model_version'] as String?,
      );

  Map<String, dynamic> toJson() => {
        'home': home,
        'draw': draw,
        'away': away,
        'model_version': modelVersion,
      };
}

class OverUnder25Market {
  const OverUnder25Market({required this.over, required this.under});

  final double over;
  final double under;

  factory OverUnder25Market.fromJson(Map<String, dynamic> json) => OverUnder25Market(
        over: (json['over_2_5'] as num).toDouble(),
        under: (json['under_2_5'] as num).toDouble(),
      );

  Map<String, dynamic> toJson() => {'over_2_5': over, 'under_2_5': under};
}

class BttsMarket {
  const BttsMarket({required this.yes, required this.no});

  final double yes;
  final double no;

  factory BttsMarket.fromJson(Map<String, dynamic> json) => BttsMarket(
        yes: (json['yes'] as num).toDouble(),
        no: (json['no'] as num).toDouble(),
      );

  Map<String, dynamic> toJson() => {'yes': yes, 'no': no};
}

class GoalsMarket {
  const GoalsMarket({this.overUnder25, this.btts});

  final OverUnder25Market? overUnder25;
  final BttsMarket? btts;

  factory GoalsMarket.fromJson(Map<String, dynamic> json) => GoalsMarket(
        overUnder25: json['over_under_2_5'] != null
            ? OverUnder25Market.fromJson(json['over_under_2_5'] as Map<String, dynamic>)
            : null,
        btts: json['btts'] != null
            ? BttsMarket.fromJson(json['btts'] as Map<String, dynamic>)
            : null,
      );

  Map<String, dynamic> toJson() => {
        'over_under_2_5': overUnder25?.toJson(),
        'btts': btts?.toJson(),
      };
}

class Markets {
  const Markets({this.w1x2, this.goals, this.metrics});

  final W1X2Market? w1x2;
  final GoalsMarket? goals;
  final MatchMetrics? metrics;

  factory Markets.fromJson(Map<String, dynamic> json) => Markets(
        w1x2: json['w1x2'] != null
            ? W1X2Market.fromJson(json['w1x2'] as Map<String, dynamic>)
            : null,
        goals: json['goals'] != null
            ? GoalsMarket.fromJson(json['goals'] as Map<String, dynamic>)
            : null,
        metrics: json['metrics'] != null
            ? MatchMetrics.fromJson(json['metrics'] as Map<String, dynamic>)
            : null,
      );

  Map<String, dynamic> toJson() => {
        'w1x2': w1x2?.toJson(),
        'goals': goals?.toJson(),
        'metrics': metrics?.toJson(),
      };
}

class FixtureResponse {
  const FixtureResponse({
    required this.id,
    required this.externalId,
    required this.competition,
    required this.season,
    required this.homeTeam,
    required this.awayTeam,
    required this.kickoffTime,
    required this.status,
    this.statusShort,
    this.venue,
    this.homeGoals,
    this.awayGoals,
    this.markets,
  });

  final int id;
  final int externalId;
  final CompetitionInfo competition;
  final SeasonInfo season;
  final TeamInfo homeTeam;
  final TeamInfo awayTeam;
  final DateTime kickoffTime;
  final String status;
  final String? statusShort;
  final String? venue;
  final int? homeGoals;
  final int? awayGoals;
  final Markets? markets;

  factory FixtureResponse.fromJson(Map<String, dynamic> json) => FixtureResponse(
        id: json['id'] as int,
        externalId: json['external_id'] as int,
        competition: CompetitionInfo.fromJson(json['competition'] as Map<String, dynamic>),
        season: SeasonInfo.fromJson(json['season'] as Map<String, dynamic>),
        homeTeam: TeamInfo.fromJson(json['home_team'] as Map<String, dynamic>),
        awayTeam: TeamInfo.fromJson(json['away_team'] as Map<String, dynamic>),
        kickoffTime: DateTime.parse(json['kickoff_time'] as String),
        status: json['status'] as String,
        statusShort: json['status_short'] as String?,
        venue: json['venue'] as String?,
        homeGoals: json['home_goals'] as int?,
        awayGoals: json['away_goals'] as int?,
        markets: json['markets'] != null
            ? Markets.fromJson(json['markets'] as Map<String, dynamic>)
            : null,
      );

  Map<String, dynamic> toJson() => {
        'id': id,
        'external_id': externalId,
        'competition': competition.toJson(),
        'season': season.toJson(),
        'home_team': homeTeam.toJson(),
        'away_team': awayTeam.toJson(),
        'kickoff_time': kickoffTime.toIso8601String(),
        'status': status,
        'status_short': statusShort,
        'venue': venue,
        'home_goals': homeGoals,
        'away_goals': awayGoals,
        'markets': markets?.toJson(),
      };
}

class PaginatedFixtures {
  const PaginatedFixtures({
    required this.items,
    required this.limit,
    required this.offset,
    required this.total,
  });

  final List<FixtureResponse> items;
  final int limit;
  final int offset;
  final int total;

  /// Fixtures grouped by competition id (preserving insertion order).
  Map<int, List<FixtureResponse>> groupedByCompetition() {
    final map = <int, List<FixtureResponse>>{};
    for (final f in items) {
      map.putIfAbsent(f.competition.id, () => []).add(f);
    }
    return map;
  }

  factory PaginatedFixtures.fromJson(Map<String, dynamic> json) => PaginatedFixtures(
        items: (json['items'] as List)
            .map((e) => FixtureResponse.fromJson(e as Map<String, dynamic>))
            .toList(),
        limit: json['limit'] as int,
        offset: json['offset'] as int,
        total: json['total'] as int,
      );

  Map<String, dynamic> toJson() => {
        'items': items.map((e) => e.toJson()).toList(),
        'limit': limit,
        'offset': offset,
        'total': total,
      };
}