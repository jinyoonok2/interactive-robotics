"""The native target_grasp composite ranking, exposed for bounded retries."""
def ranked_indices(scores, distances, max_distance, threshold=.5, score_weight=.5):
    if max_distance<=0 or len(scores)!=len(distances):
        raise ValueError('Invalid native grasp ranking inputs')
    evaluated=[(i,score_weight*float(s)+(1-score_weight)*(1-float(d)/max_distance))
               for i,(s,d) in enumerate(zip(scores,distances))]
    return sorted([(i,v) for i,v in evaluated if v>threshold],key=lambda row:row[1],reverse=True)
