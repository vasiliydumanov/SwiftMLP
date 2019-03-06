//
//  Loss.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 3/1/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix_ios

open class Loss {
    public func evaluate(y: matrix, yPred: matrix) -> vector {
        preconditionFailure("Subclass must override this method.")
    }
    
    public func backprop(y: matrix, yPred: matrix) -> matrix {
        preconditionFailure("Subclass must override this method.")
    }
}
