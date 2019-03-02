//
//  SGDOptimizer.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 3/1/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix_ios

public final class SGDOptimizer : Optimizer {
    public override func optimizeGradients(for layer: LayerWithParameters, epoch: Int) {
        let meanGrads = layer.gradients.map { grads in
            learningRate * grads.reduce(into: zeros_like(grads[0]), { agg, grad in agg = agg + grad }) / Double(grads.count)
        }
        layer.apply(gradients: meanGrads)
    }
}
