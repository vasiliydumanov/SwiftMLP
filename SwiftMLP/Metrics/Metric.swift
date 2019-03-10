//
//  Metric.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 3/1/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix

open class Metric {
    open var trainLogKey: LogKey {
        preconditionFailure("Subclass must override this property.")
    }
    open var valLogKey: LogKey {
        preconditionFailure("Subclass must override this property.")
    }
    
    open var name: String {
        preconditionFailure("Subclass must override this property.")
    }
    
    public init() {}
    
    open func evaluate(y: matrix, yPred: matrix) -> Double {
        preconditionFailure("Subclass must override this method.")
    }
}
