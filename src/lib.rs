//! plato-tile-api — Tile API with no external dependencies
//!
//! Process, search, and manage tiles via a simple stateful API.
//! Compatible with Oracle1's port 8847 PLATO server wire format.
//!
//! This crate provides the core API logic. For HTTP, see plato-tile-client.

/// A tile submitted for processing.
#[derive(Debug, Clone)]
pub struct TileInput {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub domain: String,
    pub confidence: f64,
}

/// A scored and ranked tile result.
#[derive(Debug, Clone)]
pub struct TileResult {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub domain: String,
    pub confidence: f64,
    pub score: f64,
    pub rank: usize,
}

/// A rejected tile with reasons.
#[derive(Debug, Clone)]
pub struct Rejection {
    pub id: String,
    pub reasons: Vec<String>,
}

/// API statistics.
#[derive(Debug, Clone)]
pub struct ApiStats {
    pub total_stored: usize,
    pub total_processed: usize,
    pub total_rejected: usize,
    pub pass_rate: f64,
    pub domains: std::collections::HashMap<String, usize>,
}

/// Full process response.
#[derive(Debug, Clone)]
pub struct ProcessResponse {
    pub accepted: Vec<TileResult>,
    pub rejected: Vec<Rejection>,
    pub ranked: Vec<TileResult>,
    pub stats: ApiStats,
}

/// Search response.
#[derive(Debug, Clone)]
pub struct SearchResponse {
    pub query: String,
    pub results: Vec<TileResult>,
    pub count: usize,
}

/// Configuration for the API.
#[derive(Debug, Clone)]
pub struct ApiConfig {
    pub min_confidence: f64,
    pub min_content_len: usize,
    pub max_ranked: usize,
    pub keyword_gate: f64,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            min_content_len: 10,
            max_ranked: 20,
            keyword_gate: 0.01,
        }
    }
}

/// The tile API.
pub struct TileApi {
    store: Vec<TileInput>,
    config: ApiConfig,
    total_processed: usize,
    total_rejected: usize,
    domains: std::collections::HashMap<String, usize>,
}

impl TileApi {
    pub fn new() -> Self {
        Self::with_config(ApiConfig::default())
    }

    pub fn with_config(config: ApiConfig) -> Self {
        Self {
            store: Vec::new(),
            config,
            total_processed: 0,
            total_rejected: 0,
            domains: std::collections::HashMap::new(),
        }
    }

    /// Process a batch of tiles: validate → score → store → rank.
    pub fn process(&mut self, tiles: &[TileInput], query: &str) -> ProcessResponse {
        self.total_processed += tiles.len();
        let mut accepted: Vec<TileResult> = Vec::new();
        let mut rejected: Vec<Rejection> = Vec::new();

        for tile in tiles {
            let mut issues: Vec<String> = Vec::new();
            if tile.confidence < self.config.min_confidence {
                issues.push("low confidence".into());
            }
            if tile.question.len() < self.config.min_content_len {
                issues.push("question too short".into());
            }
            if tile.answer.len() < self.config.min_content_len {
                issues.push("answer too short".into());
            }
            if tile.domain.is_empty() {
                issues.push("empty domain".into());
            }

            if !issues.is_empty() {
                self.total_rejected += 1;
                rejected.push(Rejection {
                    id: tile.id.clone(),
                    reasons: issues,
                });
                continue;
            }

            let kw = jaccard_words(&tile.question, &tile.answer, query);
            let score = if kw < self.config.keyword_gate {
                0.0
            } else {
                kw * 0.30 + tile.confidence * 0.25 + 0.8 * 0.20 + 1.0 * 0.15 + 0.10
            };

            accepted.push(TileResult {
                id: tile.id.clone(),
                question: tile.question.clone(),
                answer: tile.answer.clone(),
                domain: tile.domain.clone(),
                confidence: tile.confidence,
                score,
                rank: 0,
            });

            self.store.push(tile.clone());
            *self.domains.entry(tile.domain.clone()).or_insert(0) += 1;
        }

        // Rank
        let mut ranked: Vec<TileResult> = accepted
            .iter()
            .filter(|t| t.score > 0.0)
            .cloned()
            .collect();
        ranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, t) in ranked.iter_mut().enumerate() {
            t.rank = i + 1;
        }
        ranked.truncate(self.config.max_ranked);

        let pass_rate = if self.total_processed == 0 {
            0.0
        } else {
            (self.total_processed - self.total_rejected) as f64 / self.total_processed as f64
        };

        ProcessResponse {
            accepted: accepted.clone(),
            rejected: rejected.clone(),
            ranked: ranked.clone(),
            stats: ApiStats {
                total_stored: self.store.len(),
                total_processed: self.total_processed,
                total_rejected: self.total_rejected,
                pass_rate,
                domains: self.domains.clone(),
            },
        }
    }

    /// Search stored tiles by query.
    pub fn search(&self, query: &str, limit: usize) -> SearchResponse {
        let mut scored: Vec<TileResult> = self
            .store
            .iter()
            .map(|t| {
                let kw = jaccard_words(&t.question, &t.answer, query);
                TileResult {
                    id: t.id.clone(),
                    question: t.question.clone(),
                    answer: t.answer.clone(),
                    domain: t.domain.clone(),
                    confidence: t.confidence,
                    score: kw,
                    rank: 0,
                }
            })
            .filter(|t| t.score > 0.01)
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(limit);
        for (i, t) in scored.iter_mut().enumerate() {
            t.rank = i + 1;
        }
        SearchResponse {
            query: query.into(),
            count: scored.len(),
            results: scored,
        }
    }

    /// Get current statistics.
    pub fn stats(&self) -> ApiStats {
        let pass_rate = if self.total_processed == 0 {
            0.0
        } else {
            (self.total_processed - self.total_rejected) as f64 / self.total_processed as f64
        };
        ApiStats {
            total_stored: self.store.len(),
            total_processed: self.total_processed,
            total_rejected: self.total_rejected,
            pass_rate,
            domains: self.domains.clone(),
        }
    }

    /// Number of stored tiles.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
}

fn jaccard_words(question: &str, answer: &str, query: &str) -> f64 {
    let q_set: std::collections::HashSet<&str> = query.split_whitespace().collect();
    let t_set: std::collections::HashSet<&str> = question
        .split_whitespace()
        .chain(answer.split_whitespace())
        .collect();
    if q_set.is_empty() || t_set.is_empty() {
        return 0.0;
    }
    let inter: usize = q_set.intersection(&t_set).count();
    inter as f64 / q_set.len().max(t_set.len()) as f64
}

fn tile_in(id: &str, q: &str, a: &str, d: &str, c: f64) -> TileInput {
    TileInput {
        id: id.into(),
        question: q.into(),
        answer: a.into(),
        domain: d.into(),
        confidence: c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accepts_valid_tile() {
        let mut api = TileApi::new();
        let tiles = vec![tile_in(
            "t1",
            "What is constraint theory?",
            "Geometric snapping for exact computation.",
            "ct",
            0.9,
        )];
        let r = api.process(&tiles, "constraint theory");
        assert_eq!(r.accepted.len(), 1);
        assert_eq!(r.rejected.len(), 0);
        assert_eq!(r.stats.total_stored, 1);
    }

    #[test]
    fn test_rejects_low_confidence() {
        let mut api = TileApi::new();
        let r = api.process(
            &[tile_in("t1", "Valid question content here", "Valid answer content too", "x", 0.1)],
            "test",
        );
        assert_eq!(r.rejected.len(), 1);
        assert_eq!(r.accepted.len(), 0);
        assert!(r.rejected[0].reasons.iter().any(|s| s == "low confidence"));
    }

    #[test]
    fn test_rejects_short_question() {
        let mut api = TileApi::new();
        let r = api.process(
            &[tile_in("t1", "Short", "Valid answer content here", "d", 0.9)],
            "test",
        );
        assert_eq!(r.rejected.len(), 1);
        assert!(r.rejected[0].reasons.iter().any(|s| s == "question too short"));
    }

    #[test]
    fn test_rejects_empty_domain() {
        let mut api = TileApi::new();
        let r = api.process(
            &[tile_in("t1", "Valid question content here", "Valid answer content too", "", 0.9)],
            "test",
        );
        assert_eq!(r.rejected.len(), 1);
        assert!(r.rejected[0].reasons.iter().any(|s| s == "empty domain"));
    }

    #[test]
    fn test_ranking_best_first() {
        let mut api = TileApi::new();
        let tiles = vec![
            tile_in("t1", "What is flux?", "Bytecode runtime.", "flux", 0.8),
            tile_in("t2", "What is fishing?", "Catching fish.", "fish", 0.9),
            tile_in("t3", "What is constraint theory?", "Geometric snapping.", "ct", 0.9),
        ];
        let r = api.process(&tiles, "constraint theory geometric");
        assert!(r.ranked.len() >= 1);
        assert_eq!(r.ranked[0].id, "t3");
        assert_eq!(r.ranked[0].rank, 1);
    }

    #[test]
    fn test_keyword_gating_no_match() {
        let mut api = TileApi::new();
        let r = api.process(
            &[tile_in("t1", "What is quantum computing?", "Qubits and superposition states.", "physics", 0.9)],
            "fishing boats ocean",
        );
        assert_eq!(r.ranked.len(), 0);
        // Tile still accepted and stored
        assert_eq!(r.accepted.len(), 1);
        assert_eq!(api.len(), 1);
    }

    #[test]
    fn test_search_returns_relevant() {
        let mut api = TileApi::new();
        api.process(
            &[tile_in("t1", "What is deadband?", "P0 P1 P2 priority protocol.", "deadband", 0.9)],
            "test",
        );
        api.process(
            &[tile_in("t2", "What is flux?", "Bytecode runtime engine.", "flux", 0.8)],
            "test",
        );
        let r = api.search("deadband protocol priority", 10);
        assert_eq!(r.count, 1);
        assert_eq!(r.results[0].id, "t1");
    }

    #[test]
    fn test_stats_accumulate_across_calls() {
        let mut api = TileApi::new();
        api.process(
            &[tile_in("t1", "Valid question content here", "Valid answer content too", "plato", 0.9)],
            "test",
        );
        api.process(&[tile_in("t2", "Short", "Short", "", 0.1)], "test");
        let s = api.stats();
        assert_eq!(s.total_processed, 2);
        assert_eq!(s.total_rejected, 1);
        assert_eq!(s.total_stored, 1);
        assert_eq!(*s.domains.get("plato").unwrap_or(&0), 1);
    }

    #[test]
    fn test_batch_10_tiles() {
        let mut api = TileApi::new();
        let tiles: Vec<TileInput> = (0..10)
            .map(|i| {
                tile_in(
                    &format!("t{}", i),
                    &format!("Question {} about PLATO tiles and rooms", i),
                    &format!("Answer {} about PLATO framework architecture", i),
                    "plato",
                    0.5 + (i as f64) * 0.05,
                )
            })
            .collect();
        let r = api.process(&tiles, "PLATO tiles rooms framework");
        assert_eq!(r.accepted.len(), 10);
        assert!(r.stats.pass_rate > 0.99);
    }

    #[test]
    fn test_empty_process() {
        let mut api = TileApi::new();
        let r = api.process(&[], "test");
        assert_eq!(r.accepted.len(), 0);
        assert_eq!(r.rejected.len(), 0);
        assert_eq!(r.stats.pass_rate, 0.0);
    }

    #[test]
    fn test_empty_search() {
        let api = TileApi::new();
        let r = api.search("test", 10);
        assert_eq!(r.count, 0);
    }

    #[test]
    fn test_pass_rate_calculation() {
        let mut api = TileApi::new();
        api.process(
            &[
                tile_in("t1", "Valid question content here", "Valid answer content too", "d", 0.9),
                tile_in("t2", "Valid question content here", "Valid answer content too", "d", 0.9),
                tile_in("t3", "Short", "Short", "", 0.1),
            ],
            "test",
        );
        let s = api.stats();
        assert!((s.pass_rate - (2.0 / 3.0)).abs() < 0.001);
    }

    #[test]
    fn test_custom_config() {
        let cfg = ApiConfig {
            min_confidence: 0.8,
            min_content_len: 20,
            max_ranked: 1,
            keyword_gate: 0.5,
        };
        let mut api = TileApi::with_config(cfg);
        let tiles = vec![
            tile_in("t1", "Valid question content here", "Valid answer content too", "d", 0.9),
            tile_in("t2", "Valid question content here", "Valid answer content too", "d", 0.7),
        ];
        let r = api.process(&tiles, "test");
        assert_eq!(r.accepted.len(), 1);
        assert_eq!(r.rejected.len(), 1);
        assert!(r.ranked.len() <= 1);
    }

    #[test]
    fn test_is_empty() {
        let api = TileApi::new();
        assert!(api.is_empty());
    }

    #[test]
    fn test_len_tracks_stored() {
        let mut api = TileApi::new();
        assert_eq!(api.len(), 0);
        api.process(
            &[tile_in("t1", "Valid question content here", "Valid answer content too", "d", 0.9)],
            "test",
        );
        assert_eq!(api.len(), 1);
        // Rejected tiles don't count
        api.process(&[tile_in("t2", "Short", "Short", "", 0.1)], "test");
        assert_eq!(api.len(), 1);
    }
}
