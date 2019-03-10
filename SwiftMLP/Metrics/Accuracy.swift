//
//  Accuracy.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 3/1/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix

public final class Accuracy : Metric {
    public override var trainLogKey: LogKey {
        return .trainAccuracy
    }
    public override var valLogKey: LogKey {
        return .valAccuracy
    }
    
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

public extension LogKey {
    public static let trainAccuracy = LogKey(rawValue: "train_acc")
    public static let valAccuracy = LogKey(rawValue: "val_acc")
}
