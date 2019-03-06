//
//  Callback.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 3/3/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation


open class Callback {
    public enum Priority : Int {
        case begin = 0
        case middle = 1
        case end = 2
    }
    
    public var _model: Model!
    
    open var priority: Priority {
        preconditionFailure("Subclass must override this property.")
    }
    
    open func onTrainBegin() {
    }
    
    open func onTrainEnd(log: inout Log) {
    }
    
    open func onEpochBegin(epoch: Int, log: inout Log) {
    }
    
    open func onEpochEnd(epoch: Int, log: inout Log) -> Bool {
        return true
    }
    
    open func onBatchBegin() {
    }
    
    open func onBatchEnd() -> Bool {
        return true
    }
}


