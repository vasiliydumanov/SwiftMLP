//
//  Helpers.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 3/1/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix_ios


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
