//
//  Metric.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 3/1/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix_ios

open class Metric {
    public var trainLogKey: LogKey {
        preconditionFailure("Subclass must override this property.")
    }
    public var valLogKey: LogKey {
        preconditionFailure("Subclass must override this property.")
    }
    
    public var name: String {
        preconditionFailure("Subclass must override this property.")
    }
    
    public func evaluate(y: matrix, yPred: matrix) -> Double {
        preconditionFailure("Subclass must override this method.")
    }
}
