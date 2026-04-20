//! Stateful tile API — process, search, score, rank in one interface.
//! Part of the PLATO framework.

/// A knowledge tile.
#[derive(Debug, Clone)]
pub struct Tile {
    pub content: String,
    pub domain: String,
    pub confidence: f64,
}

/// Result of processing a tile through the pipeline.
#[derive(Debug, Clone)]
pub struct TilePipelineResult {
    pub tile: Tile,
    pub passed: bool,
    pub score: f64,
    pub reason: String,
}

/// A tile with its computed score.
#[derive(Debug, Clone)]
pub struct ScoredTile {
    pub tile: Tile,
    pub score: f64,
}

/// API usage statistics.
#[derive(Debug, Clone)]
pub struct ApiStats {
    pub total: usize,
    pub avg_confidence: f64,
    pub searches: usize,
}

/// Stateful tile API — the "one interface" for tile operations.
pub struct TileAPI {
    tiles: Vec<Tile>,
    searches: usize,
}

impl TileAPI {
    pub fn new() -> Self {
        Self { tiles: Vec::new(), searches: 0 }
    }

    /// Process tiles: validate, score, store passed tiles.
    pub fn process(&mut self, tiles: Vec<Tile>) -> Vec<TilePipelineResult> {
        tiles.into_iter().map(|tile| {
            let content_len = tile.content.len();
            let conf = tile.confidence;
            let passed = content_len >= 10 && conf >= 0.3;
            let score = if passed {
                conf * 0.3 + 1.0 * 0.7
            } else {
                conf * 0.1
            };
            let reason = if passed {
                String::from("passed")
            } else if content_len < 10 {
                format!("content too short ({} chars)", content_len)
            } else {
                format!("confidence too low ({:.2})", conf)
            };
            if passed {
                self.tiles.push(tile.clone());
            }
            TilePipelineResult { tile, passed, score, reason }
        }).collect()
    }

    /// Search tiles by keyword overlap with Jaccard-like ranking.
    pub fn search(&self, query: &str, top_n: usize) -> Vec<ScoredTile> {
        let q_words: std::collections::HashSet<String> =
            query.to_lowercase().split_whitespace().map(String::from).collect();
        if q_words.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<ScoredTile> = self.tiles.iter().map(|tile| {
            let c_words: std::collections::HashSet<String> =
                tile.content.to_lowercase().split_whitespace().map(String::from).collect();
            let union = q_words.union(&c_words).count();
            let overlap = q_words.intersection(&c_words).count();
            let jaccard = overlap as f64 / union.max(1) as f64;
            let domain_bonus = if query.to_lowercase().contains(&tile.domain.to_lowercase()) { 0.3 } else { 0.0 };
            let score = jaccard * 0.5 + tile.confidence * 0.2 + domain_bonus;
            ScoredTile { tile: tile.clone(), score }
        }).filter(|s| s.score > 0.01)
          .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_n);
        scored
    }

    /// Return API statistics.
    pub fn stats(&self) -> ApiStats {
        let avg = if self.tiles.is_empty() {
            0.0
        } else {
            self.tiles.iter().map(|t| t.confidence).sum::<f64>() / self.tiles.len() as f64
        };
        ApiStats { total: self.tiles.len(), avg_confidence: avg, searches: self.searches }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tile(content: &str, domain: &str, confidence: f64) -> Tile {
        Tile { content: content.to_string(), domain: domain.to_string(), confidence }
    }

    #[test]
    fn test_process_valid_tile() {
        let mut api = TileAPI::new();
        let results = api.process(vec![sample_tile("Pythagorean triples snap to exact coordinates", "math", 0.9)]);
        assert_eq!(results.len(), 1);
        assert!(results[0].passed);
        assert!(results[0].score > 0.5);
        assert_eq!(api.stats().total, 1);
    }

    #[test]
    fn test_process_rejects_short_content() {
        let mut api = TileAPI::new();
        let results = api.process(vec![sample_tile("short", "test", 0.9)]);
        assert!(!results[0].passed);
        assert!(results[0].reason.contains("short"));
        assert_eq!(api.stats().total, 0);
    }

    #[test]
    fn test_process_rejects_low_confidence() {
        let mut api = TileAPI::new();
        let results = api.process(vec![sample_tile("A reasonably long content string here", "test", 0.1)]);
        assert!(!results[0].passed);
        assert!(results[0].reason.contains("confidence"));
    }

    #[test]
    fn test_search_keyword_match() {
        let mut api = TileAPI::new();
        api.process(vec![
            sample_tile("Pythagorean triples snap to exact coordinates", "math", 0.9),
            sample_tile("Deadband Protocol enforces strict priority ordering", "governance", 0.8),
        ]);
        let results = api.search("Pythagorean exact coordinates", 5);
        assert!(results.len() >= 1);
        assert_eq!(results[0].tile.domain, "math");
    }

    #[test]
    fn test_search_domain_bonus() {
        let mut api = TileAPI::new();
        api.process(vec![
            sample_tile("Some generic content about things and stuff", "math", 0.7),
        ]);
        let results = api.search("math domain", 5);
        assert!(results.len() >= 1);
        assert!(results[0].score > 0.2); // domain bonus contributes
    }

    #[test]
    fn test_search_top_n() {
        let mut api = TileAPI::new();
        for i in 0..20 {
            api.process(vec![sample_tile(&format!("Tile number {} with some content here", i), "test", 0.5 + i as f64 * 0.02)]);
        }
        let results = api.search("tile content", 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_stats() {
        let mut api = TileAPI::new();
        api.process(vec![
            sample_tile("First tile content here", "a", 0.8),
            sample_tile("Second tile content here", "b", 0.6),
        ]);
        let stats = api.stats();
        assert_eq!(stats.total, 2);
        assert!((stats.avg_confidence - 0.7).abs() < 0.01);
        assert_eq!(stats.searches, 0);
    }

    #[test]
    fn test_empty_api() {
        let api = TileAPI::new();
        assert_eq!(api.stats().total, 0);
        assert_eq!(api.search("anything", 10).len(), 0);
    }

    #[test]
    fn test_process_batch() {
        let mut api = TileAPI::new();
        let tiles = vec![
            sample_tile("Valid content for processing", "test", 0.9),
            sample_tile("x", "test", 0.9),          // too short
            sample_tile("Another valid tile here", "test", 0.1), // too low conf
            sample_tile("Yet another valid tile", "test", 0.7),
        ];
        let results = api.process(tiles);
        assert_eq!(results.len(), 4);
        assert_eq!(results.iter().filter(|r| r.passed).count(), 2);
        assert_eq!(api.stats().total, 2);
    }

    #[test]
    fn test_search_empty_query() {
        let mut api = TileAPI::new();
        api.process(vec![sample_tile("Some content", "test", 0.9)]);
        assert_eq!(api.search("", 10).len(), 0);
    }
}
