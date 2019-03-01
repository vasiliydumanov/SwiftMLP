//
//  Metric.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 3/1/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix_ios

public class Metric {
    public var name: String {
        preconditionFailure("Subclass must override this property.")
    }
    
    public func evaluate(y: matrix, yPred: matrix) -> Double {
        preconditionFailure("Subclass must override this method.")
    }
}
