.. currentmodule:: starkman_thesis

*************
Term Glossary
*************

.. glossary::

   number
      Any numeric type. eg float or int or any of the :class:`~numpy.number`.

   -like
      Used to indicate on object of that type or that can instantiate the
      type, eg :class:`~astropy.units.Quantity`-like includes ``"2 * u.km"``
      because ``astropy.units.Quantity("2 * u.km")`` works.

   unit-like
      Must be an :class:`~astropy.units.UnitBase` (subclass) instance or a
      string or other instance parseable by :class:`~astropy.units.Unit`.

   quantity-like
      Must be an :class:`~astropy.units.Quantity` (or subclass) instance or a
      string parseable by :class:`~astropy.units.Quantity`. Note that the
      interpretation of units in strings depends on the class --
      ``Quantity("180d")`` is 180 days, while ``Angle("180d")`` is 180 degrees
      -- so make sure the string parses as intended for ``Quantity``.

   angle-like
      :term:`Quantity-like`, but interpreted by an angular
      :class:`~astropy.units.SpecificTypeQuantity`, like
      :class:`~astropy.coordinates.Angle` or
      :class:~astropy.coordinates.Longitude` or
      :class:~astropy.coordinates.Latitude`. Note that the interpretation of
      units in strings depends on the class -- ``Quantity("180d")`` is 180
      days, while ``Angle("180d")`` is 180 degrees -- so make sure the
      string parses as intended for ``Angle``.

   length-like
      :term:`Quantity-like`, but interpretable by
      :class:`~astropy.coordinates.Distance`.

   frame-like
      A :class:`~astropy.coordinates.BaseCoordinateFrame` subclass instance or
      a string that can be converted to a Frame by
      :class:`~astropy.coordinates.sky_coordinate_parsers._get_frame_class`.

   coord-like
      A Coordinate-type object such as a
      :class:`~astropy.coordinates.BaseCoordinateFrame` subclass instance or a
      :class:`~astropy.coordinates.SkyCoord` (or subclass) instance.

   table-like
      An astropy :class:`~astropy.table.Table` or any object that can
      initialize one. Anything marked as Table-like will be processed through
      a :class:`~astropy.table.Table`.

   time-like
      :class:`~astropy.time.Time` or any valid initializer.


***************************
Optional Packages' Glossary
***************************

.. glossary::

   color
      Any valid Matplotlib color.
