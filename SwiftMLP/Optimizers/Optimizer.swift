//
//  Optimizer.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 3/1/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix_ios

open class Optimizer {
    public var learningRate: Double
    
    public init(learningRate: Double = 0.0001) {
        self.learningRate = learningRate
    }
    
    open func optimizeGradients(for layer: LayerWithParameters, epoch: Int) {
        preconditionFailure("Subclass must override this method.")
    }
}
