"""
Point-In-Time AS-OF Joiner
Guarantees temporal consistency and eliminates data leakage in offline model training datasets.
"""

from typing import List, Dict, Any

class PointInTimeJoiner:
    """Performs temporal AS-OF joins matching event timestamps to most recent prior feature state."""
    def __init__(self, max_lookback_seconds: int = 86400 * 30):
        self.max_lookback_seconds = max_lookback_seconds

    def asof_join(
        self,
        observation_events: List[Dict[str, Any]],
        feature_events: List[Dict[str, Any]],
        entity_key: str,
        timestamp_key: str
    ) -> List[Dict[str, Any]]:
        # Sort feature events chronologically
        sorted_features = sorted(feature_events, key=lambda x: x[timestamp_key])
        joined_results = []
        
        for obs in observation_events:
            obs_time = obs[timestamp_key]
            obs_entity = obs[entity_key]
            
            # Find latest feature where feature_time <= obs_time
            matched_feature = None
            for feat in sorted_features:
                if feat[entity_key] == obs_entity and feat[timestamp_key] <= obs_time:
                    matched_feature = feat
                elif feat[timestamp_key] > obs_time:
                    break
                    
            record = {**obs}
            if matched_feature:
                for k, v in matched_feature.items():
                    if k not in (entity_key, timestamp_key):
                        record[k] = v
            joined_results.append(record)
            
        return joined_results
