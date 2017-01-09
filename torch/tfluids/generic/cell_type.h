// Copyright 2016 Google Inc, NYU.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// These are the same enum values used in Manta. We can't include grid.h
// from Manta without pulling in the entire library, so we'll just redefine
// them here.
enum CellType {
    TypeNone = 0,
    TypeFluid = 1,
    TypeObstacle = 2,
    TypeEmpty = 4,
    TypeInflow = 8,
    TypeOutflow = 16,
    TypeOpen = 32,
    TypeStick = 128,
    TypeReserved = 256,
    TypeZeroPressure = (1<<15)
};

