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
    public override func optimizeGradients(for layer: LayerWithParameters) {
        layer.apply(gradients: layer.gradients)
    }
}
