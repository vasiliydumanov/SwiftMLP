//
//  SoftmaxCrossentropy.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 3/1/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix

public final class SoftmaxCrossentropy : Loss {
    public override init() { super.init() }
    
    public override func evaluate(y: matrix, yPred: matrix) -> vector {
        return -log(sum(yPred * y, axis: 1))
    }
    
    public override func backprop(y: matrix, yPred: matrix) -> matrix {
        fatalError("This function must not be used")
    }
}
