//
//  AdamOptimizer.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 3/2/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix

public final class AdamOptimizer : Optimizer {
    public let beta1: Double
    public let beta2: Double
    public let epsilon: Double
    
    public init(learningRate: Double = 1e-4, beta1: Double = 0.9, beta2: Double = 0.999, epsilon: Double = 1e-8) {
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        super.init(learningRate: learningRate)
    }
    
    public override func optimizeGradients(for layer: LayerWithParameters, epoch: Int) {
        let meanGrads = layer.gradients.map { grads in
            grads.reduce(into: zeros_like(grads[0]), { agg, grad in agg = agg + grad }) / Double(grads.count)
        }
        
        var ms: [matrix]
        var vs: [matrix]
        if layer.states.isEmpty {
            ms = meanGrads.map { grad in zeros_like(grad) }
            vs = meanGrads.map { grad in zeros_like(grad) }
        } else {
            ms = layer.states[0]
            vs = layer.states[1]
        }
        
        for i in 0..<ms.count {
            ms[i] = beta1 * ms[i] + (1 - beta1) * meanGrads[i]
        }
        for i in 0..<vs.count {
            vs[i] = beta2 * vs[i] + (1 - beta2) * pow(meanGrads[i], power: 2.0)
        }
        layer.states = [ms, vs]
        
        let beta1denom = 1 - pow(beta1, Double(epoch))
        let beta2denom = 1 - pow(beta2, Double(epoch))
        
        let resGrads = zip(ms, vs).map { m, v in
            learningRate * (m / beta1denom) / (pow(v / beta2denom, power: 0.5) + epsilon)
        }
        
        layer.apply(gradients: resGrads)
    }
}

