

#include <vector>

// MAPPING #0
//
// MIT SCENE PARSING DATASET 151 categories --> 6 categories
// {
//   0 : nothing
//   1 : buildings
//   2 : sky
//   3 : background (generic stuff)
//   4 : person, animals
//   5 : vehicles
// }

static const std::vector<unsigned char> om0000_mit_151_007 = {
    0,  // : 0 = Nothing
    3,  // : 1 = wall
    1,  // : 2 = building,
    2,  // : 3 = sky
    3,  // : 4 = floor,
    3,  // : 5 = tree
    3,  // : 6 = ceiling
    3,  // : 7 = road,
    3,  // : 8 = bed
    3,  // : 9 = windowpane,
    3,  // : 10 = grass
    3,  // : 11 = cabinet
    3,  // : 12 = sidewalk,
    4,  // : 13 = person,
    3,  // : 14 = earth,
    3,  // : 15 = door,
    3,  // : 16 = table
    3,  // : 17 = mountain,
    3,  // : 18 = plant,
    3,  // : 19 = curtain,
    3,  // : 20 = chair
    5,  // : 21 = car,
    3,  // : 22 = water
    3,  // : 23 = painting,
    3,  // : 24 = sofa,
    3,  // : 25 = shelf
    1,  // : 26 = house
    3,  // : 27 = sea
    3,  // : 28 = mirror
    3,  // : 29 = rug,
    3,  // : 30 = field
    3,  // : 31 = armchair
    3,  // : 32 = seat
    3,  // : 33 = fence,
    3,  // : 34 = desk
    3,  // : 35 = rock,
    3,  // : 36 = wardrobe,
    3,  // : 37 = lamp
    3,  // : 38 = bathtub,
    3,  // : 39 = railing,
    3,  // : 40 = cushion
    3,  // : 41 = base,
    3,  // : 42 = box
    3,  // : 43 = column,
    3,  // : 44 = signboard,
    3,  // : 45 = chest
    3,  // : 46 = counter
    3,  // : 47 = sand
    3,  // : 48 = sink
    1,  // : 49 = skyscraper
    3,  // : 50 = fireplace,
    3,  // : 51 = refrigerator,
    3,  // : 52 = grandstand,
    3,  // : 53 = path
    3,  // : 54 = stairs,
    3,  // : 55 = runway
    3,  // : 56 = case,
    3,  // : 57 = pool
    3,  // : 58 = pillow
    3,  // : 59 = screen
    3,  // : 60 = stairway,
    3,  // : 61 = river
    3,  // : 62 = bridge,
    3,  // : 63 = bookcase
    3,  // : 64 = blind,
    3,  // : 65 = coffee
    3,  // : 66 = toilet,
    3,  // : 67 = flower
    3,  // : 68 = book
    3,  // : 69 = hill
    3,  // : 70 = bench
    3,  // : 71 = countertop
    3,  // : 72 = stove,
    3,  // : 73 = palm,
    3,  // : 74 = kitchen
    3,  // : 75 = computer,
    3,  // : 76 = swivel
    3,  // : 77 = boat
    3,  // : 78 = bar
    3,  // : 79 = arcade
    3,  // : 80 = hovel,
    5,  // : 81 = bus,
    3,  // : 82 = towel
    3,  // : 83 = light,
    5,  // : 84 = truck,
    1,  // : 85 = tower
    3,  // : 86 = chandelier,
    3,  // : 87 = awning,
    3,  // : 88 = streetlight,
    3,  // : 89 = booth,
    3,  // : 90 = television
    5,  // : 91 = airplane,
    3,  // : 92 = dirt
    3,  // : 93 = apparel,
    3,  // : 94 = pole
    3,  // : 95 = land,
    3,  // : 96 = bannister,
    3,  // : 97 = escalator,
    3,  // : 98 = ottoman,
    3,  // : 99 = bottle
    3,  // : 100 = buffet,
    3,  // : 101 = poster,
    3,  // : 102 = stage
    5,  // : 103 = van
    5,  // : 104 = ship
    3,  // : 105 = fountain
    3,  // : 106 = conveyer
    3,  // : 107 = canopy
    3,  // : 108 = washer,
    3,  // : 109 = plaything,
    3,  // : 110 = swimming
    3,  // : 111 = stool
    3,  // : 112 = barrel,
    3,  // : 113 = basket,
    3,  // : 114 = waterfall,
    3,  // : 115 = tent,
    3,  // : 116 = bag
    3,  // : 117 = minibike,
    3,  // : 118 = cradle
    3,  // : 119 = oven
    3,  // : 120 = ball
    3,  // : 121 = food,
    3,  // : 122 = step,
    3,  // : 123 = tank,
    3,  // : 124 = trade
    3,  // : 125 = microwave,
    3,  // : 126 = pot,
    4,  // : 127 = animal,
    5,  // : 128 = bicycle,
    3,  // : 129 = lake
    3,  // : 130 = dishwasher,
    3,  // : 131 = screen,
    3,  // : 132 = blanket,
    3,  // : 133 = sculpture
    3,  // : 134 = hood,
    3,  // : 135 = sconce
    3,  // : 136 = vase
    3,  // : 137 = traffic
    3,  // : 138 = tray
    3,  // : 139 = ashcan,
    3,  // : 140 = fan
    3,  // : 141 = pier,
    3,  // : 142 = crt
    3,  // : 143 = plate
    3,  // : 144 = monitor,
    3,  // : 145 = bulletin
    3,  // : 146 = shower
    3,  // : 147 = radiator
    3,  // : 148 = glass,
    3,  // : 149 = clock
    3   // : 150 = flag
};


// return the mapping given its ID
static const std::vector<unsigned char> getObjectsMapping( int /*id*/ ) { return om0000_mit_151_007; }
