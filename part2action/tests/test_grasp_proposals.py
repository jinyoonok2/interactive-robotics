from diagnostics.grasp_proposals import ranked_indices


def test_native_score_distance_ranking_preserves_filter_and_stable_order():
    ranking=ranked_indices([.8,.9,.99,.1],[.1,.2,.3,.9],1.)
    assert [i for i,v in ranking]==[0,1,2]
    assert all(v>.5 for i,v in ranking)
    assert ranked_indices([0.],[1.],1.)==[]
