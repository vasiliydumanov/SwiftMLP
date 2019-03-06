//
//  Callback.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 3/3/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation


public class Callback {
    public enum Priority : Int {
        case begin = 0
        case middle = 1
        case end = 2
    }
    
    public var _model: Model!
    
    public var priority: Priority {
        preconditionFailure("Subclass must override this property.")
    }
    
    public func onTrainBegin() {
    }
    
    public func onTrainEnd(log: inout Log) {
    }
    
    public func onEpochBegin(epoch: Int, log: inout Log) {
    }
    
    public func onEpochEnd(epoch: Int, log: inout Log) -> Bool {
        return true
    }
    
    public func onBatchBegin() {
    }
    
    public func onBatchEnd() -> Bool {
        return true
    }
}


