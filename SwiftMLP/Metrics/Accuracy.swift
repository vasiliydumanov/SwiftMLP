//
//  Accuracy.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 3/1/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix_ios

public final class Accuracy : Metric {
    public override init() { super.init() }
    
    public override var name: String {
        return "accuracy"
    }
    
    public override func evaluate(y: matrix, yPred: matrix) -> Double {
        let yPredRounded = yPred == max(yPred, axis: 1).reshape((yPred.shape.0, 1))
        let yPredIsEqualToY = yPredRounded && y
        return mean(sum(yPredIsEqualToY, axis: 1))
    }
}
