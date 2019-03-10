//
//  Helpers.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 3/1/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix


public func glorotn(_ shape: (Int, Int), fanIn: Int, fanOut: Int) -> matrix {
    let variance = 2.0 / (fanIn + fanOut)
    let sigma = sqrt(variance)
    return randn(shape, mean: 0, sigma: sigma)
}

public func onehot(_ vec: vector, nClasses: Int) -> matrix {
    var mat = zeros((vec.n, nClasses))
    for i in 0..<mat.rows {
        for k in 0..<mat.columns {
            mat[i, k] = vec[i] == k ? 1 : 0
        }
    }
    return mat
}

public func metricToLogStr(_ metric: Double) -> String {
    let smallestExp: Double = 5
    let normalizer: Double = pow(10, smallestExp)
    let rndMetric = round(metric * normalizer) / normalizer
    if rndMetric != 0 {
        return String(rndMetric)
    } else {
        return String(format: "%e", rndMetric)
    }
}
